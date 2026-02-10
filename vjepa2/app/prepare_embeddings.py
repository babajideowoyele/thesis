# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Prepare VJEPA embeddings for MVFoul dataset to enable fast classifier training.

Usage:
    python app/prepare_embeddings.py --config-name embedding_prepare \
        dataset.root_dir=/path/to/mvfoul \
        model.pretrained_path=/path/to/vitl.pt \
        output.dir=./embeddings
        
    Or with defaults:
    python app/prepare_embeddings.py --config-name embedding_prepare
"""

import ast
import json
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.cluster import KMeans

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_large_rope, vit_giant_xformers_rope
from src.datasets.utils.mvfoul_translations import ActionClass, translate_annotation

class VJEPAEmbeddingExtractor:
    def __init__(self, cfg: DictConfig):
        """
        Initialize VJEPA embedding extractor from config.
        
        Args:
            cfg: Hydra config with model settings
        """
        self.device = cfg.embedding.device
        self.img_size = cfg.model.img_size
        self.model_variant = cfg.model.variant
        self.verbose = cfg.output.verbose
        self.IMAGENET_DEFAULT_MEAN: tuple[float] = ast.literal_eval(cfg.dataset.IMAGENET_DEFAULT_MEAN)
        self.IMAGENET_DEFAULT_STD: tuple[float] = ast.literal_eval(cfg.dataset.IMAGENET_DEFAULT_STD)
        
        # Initialize model
        if cfg.model.variant == "vitl":
            self.model = vit_large_rope(img_size=(cfg.model.img_size, cfg.model.img_size), num_frames=cfg.model.num_frames)
            self.num_frames = cfg.model.num_frames
        elif cfg.model.variant == "vitg":
            self.model = vit_giant_xformers_rope(img_size=(cfg.model.img_size, cfg.model.img_size), num_frames=cfg.model.num_frames)
            self.num_frames = cfg.model.num_frames
        else:
            raise ValueError(f"Unknown model variant: {cfg.model.variant}")
        
        # Load pretrained weights
        self._load_pretrained_weights(cfg.model.pretrained_path)
        self.model = self.model.to(self.device).eval()
        
        # Build transform
        self.transform = self._build_transform(cfg.model.img_size)
    
    def _load_pretrained_weights(self, model_path: str):
        """Load pretrained VJEPA weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        pretrained_dict = torch.load(model_path, weights_only=True, map_location="cpu")["encoder"]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = self.model.load_state_dict(pretrained_dict, strict=False)
        print(f"Loaded pretrained weights from {model_path}")
        print(f"Load message: {msg}")
    
    def _build_transform(self, img_size: int):
        """Build video preprocessing transform."""
        short_side_size = int(256.0 / 224 * img_size)
        eval_transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=self.IMAGENET_DEFAULT_MEAN, std=self.IMAGENET_DEFAULT_STD),
        ])
        return eval_transform
    
    def load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Load video and sample frames uniformly."""
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            # Sample frames uniformly to get self.num_frames frames
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            video = vr.get_batch(frame_indices)  # (T, H, W, C)
            
            return torch.from_numpy(video.asnumpy())  # Convert to torch.Tensor
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    @torch.no_grad()
    def extract_embeddings(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-wise embeddings from video.
        
        Args:
            video: Video array of shape (T, H, W, C)
            
        Returns:
            Embeddings of shape (T*num_patches, embed_dim)
        """
        if video is None:
            return None
        
        # Convert to torch and reorder to (T, C, H, W)
        video_tensor = video.permute(0, 3, 1, 2).float()
        
        # Apply transform
        video_tensor = self.transform(video_tensor)
        
        # Add batch dimension and move to device
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        embeddings = self.model(video_tensor)  # (B, T*num_patches, embed_dim)
        
        return embeddings[0].cpu()  # Remove batch dimension and return to CPU


class MVFoulVideoDataset(Dataset):
    """PyTorch Dataset for loading MVFoul videos in parallel."""
    
    def __init__(
        self,
        split_dir: str,
        action_dirs: list,
        annotations: dict,
        num_frames: int,
        transform,
        precomputed_frame_indices: Optional[Dict[int, np.ndarray]] = None,
        verbose: bool = False,
    ):
        self.split_dir = split_dir
        self.action_dirs = action_dirs
        self.annotations = annotations
        self.num_frames = num_frames
        self.transform = transform
        self.verbose = verbose
        self.precomputed_frame_indices = precomputed_frame_indices or {}
        
        # Build list of all video paths with metadata
        self.samples = []
        for action_idx, action_dir in enumerate(action_dirs):
            action_path = os.path.join(split_dir, action_dir)
            video_files = sorted([f for f in os.listdir(action_path) if f.endswith(".mp4")])
            
            # Get label
            annotation = annotations[str(action_idx)]
            label = translate_annotation(annotation)["type"]
            
            for angle_idx, video_file in enumerate(video_files):
                video_path = os.path.join(action_path, video_file)
                self.samples.append({
                    "video_path": video_path,
                    "label": label,
                    "action_class": ActionClass(label).name,
                    "action_dir": action_dir,
                    "video_file": video_file,
                    "angle": angle_idx,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load video
            vr = VideoReader(sample["video_path"])
            total_frames = len(vr)
            
            # Use precomputed frame indices if available, otherwise uniform sampling
            if idx in self.precomputed_frame_indices:
                frame_indices = self.precomputed_frame_indices[idx]
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            video = vr.get_batch(frame_indices)  # (T, H, W, C)
            video_tensor = torch.from_numpy(video.asnumpy())
            
            # Apply transform
            video_tensor = video_tensor.permute(0, 3, 1, 2).float()  # (T, C, H, W)
            video_tensor = self.transform(video_tensor)
            
            return {
                "video": video_tensor,
                "label": sample["label"],
                "action_class": sample["action_class"],
                "action_dir": sample["action_dir"],
                "video_file": sample["video_file"],
                "angle": sample["angle"],
                "success": True,
            }
        except Exception as e:
            if self.verbose:
                print(f"Error loading {sample['video_path']}: {e}")
            # Return None sample on failure
            return {
                "video": None,
                "label": sample["label"],
                "action_class": sample["action_class"],
                "action_dir": sample["action_dir"],
                "video_file": sample["video_file"],
                "angle": sample["angle"],
                "success": False,
            }


class MVFoulEmbeddingPreparator:
    def __init__(
        self,
        cfg: DictConfig,
        extractor: VJEPAEmbeddingExtractor,
    ):
        """
        Initialize embedding preparator for MVFoul dataset.
        
        Args:
            cfg: Hydra config
            extractor: VJEPAEmbeddingExtractor instance
        """
        self.cfg = cfg
        self.data_root = cfg.dataset.root_dir
        self.split = cfg.dataset.split
        self.extractor = extractor
        self.output_dir = Path(cfg.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = cfg.output.verbose
        
        # Load dataset metadata
        self._load_dataset_metadata()
    
    def _load_dataset_metadata(self):
        """Load MVFoul dataset metadata."""
        split_dir = os.path.join(self.data_root, self.split)
        annotations_path = os.path.join(split_dir, "annotations.json")
        
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations not found at {annotations_path}")
        
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        
        self.annotations = annotations["Actions"]
        self.split_dir = split_dir
        
        # Get action directories
        entries_unordered = os.listdir(split_dir)
        self.action_dirs = ["action_" + str(e) for e in range( len(entries_unordered) ) if os.path.isdir(os.path.join(split_dir, "action_" + str(e)))]
    
    def _precompute_frame_indices(self) -> Dict[int, np.ndarray]:
        """
        Precompute frame indices for all videos using player detection.
        Runs in main process before DataLoader workers start.
        
        Returns:
            Dictionary mapping sample index to frame indices
        """
        if not self.cfg.player_detection.enabled:
            if self.verbose:
                print("Player detection disabled, using uniform sampling")
            return {}
        
        if self.verbose:
            print("\nPrecomputing frame indices with player detection...")
        
        # Initialize detection pipeline
        try:
            from transformers import pipeline
            from PIL import Image
            
            device_config = self.cfg.player_detection.get("device", "cuda")
            device = 0 if device_config == "cuda" and torch.cuda.is_available() else -1
            
            detection_pipeline = pipeline(
                "zero-shot-object-detection",
                model=self.cfg.player_detection.model,
                device=device,
                use_fast=True if device != -1 else False,
            )
            
            device_name = "cuda" if device == 0 else "cpu"
            if self.verbose:
                print(f"Loaded detection model: {self.cfg.player_detection.model} on {device_name}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load detection model: {e}")
                print("Falling back to uniform sampling")
            return {}
        
        # Helper function for batch detection
        def batch_frames_have_players(frames: list) -> list:
            """Process multiple frames in batch for efficiency."""
            try:
                images = [Image.fromarray(frame) for frame in frames]
                # Batch process all frames at once
                results = detection_pipeline(
                    images,
                    candidate_labels=self.cfg.player_detection.queries,
                    batch_size=len(images)
                )
                # Check each frame's results
                flags = []
                for frame_results in results:
                    has_player = False
                    for detection in frame_results:
                        if detection.get("score", 0) >= self.cfg.player_detection.confidence_threshold:
                            has_player = True
                            break
                    flags.append(has_player)
                return flags
            except Exception as e:
                if self.verbose:
                    print(f"Batch detection failed: {e}")
                return [False] * len(frames)
        
        def longest_true_run(flags: list, indices: np.ndarray) -> Optional[tuple[int, int]]:
            best_start = None
            best_end = None
            best_len = 0
            current_start = None
            
            for i, flag in enumerate(flags):
                if flag and current_start is None:
                    current_start = i
                if not flag and current_start is not None:
                    current_len = i - current_start
                    if current_len > best_len:
                        best_len = current_len
                        best_start = current_start
                        best_end = i - 1
                    current_start = None
            
            if current_start is not None:
                current_len = len(indices) - current_start
                if current_len > best_len:
                    best_start = current_start
                    best_end = len(indices) - 1
            
            if best_start is None or best_end is None:
                return None
            
            return int(indices[best_start]), int(indices[best_end])
        
        # Build sample list (same as in Dataset)
        samples = []
        for action_idx, action_dir in enumerate(self.action_dirs):
            action_path = os.path.join(self.split_dir, action_dir)
            video_files = sorted([f for f in os.listdir(action_path) if f.endswith(".mp4")])
            
            annotation = self.annotations[str(action_idx)]
            label = translate_annotation(annotation)["type"]
            
            for angle_idx, video_file in enumerate(video_files):
                video_path = os.path.join(action_path, video_file)
                samples.append({
                    "video_path": video_path,
                    "label": label,
                    "action_dir": action_dir,
                    "video_file": video_file,
                })
        
        # Precompute indices for each video
        frame_indices_dict = {}
        probe_frames = self.cfg.player_detection.probe_frames
        
        for idx, sample in enumerate(tqdm(samples, desc="Precomputing frame indices")):
            try:
                vr = VideoReader(sample["video_path"])
                total_frames = len(vr)
                
                if total_frames <= 0:
                    continue
                
                # Probe frames for player detection
                probe_count = min(probe_frames, total_frames)
                probe_indices = np.linspace(0, total_frames - 1, probe_count, dtype=int)
                probe_frames_data = vr.get_batch(probe_indices).asnumpy()
                
                # Detect players in batch for efficiency
                flags = batch_frames_have_players(probe_frames_data)
                chunk = longest_true_run(flags, probe_indices)
                
                if chunk is not None:
                    chunk_start, chunk_end = chunk
                    if chunk_end > chunk_start:
                        # Sample randomly from the chunk
                        candidate_indices = np.arange(chunk_start, chunk_end + 1)
                        rng = np.random.default_rng()
                        replace = len(candidate_indices) < self.extractor.num_frames
                        selected = rng.choice(
                            candidate_indices,
                            size=self.extractor.num_frames,
                            replace=replace
                        )
                        frame_indices_dict[idx] = np.sort(selected)
                        continue
                
                # Fallback to uniform if no chunk found
                # (Don't store uniform indices, let Dataset compute them)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error precomputing indices for {sample['video_file']}: {e}")
                continue
        
        if self.verbose:
            print(f"Precomputed indices for {len(frame_indices_dict)}/{len(samples)} videos\n")
        
        return frame_indices_dict
    
    def prepare(self) -> Dict[int, Dict[str, Any]]:
        """
        Prepare embeddings for all videos in the dataset using DataLoader.
        Each video angle is treated as a separate instance.
        
        Returns:
            Dictionary mapping index to {label, features}
        """
        # Precompute frame indices with player detection (main process)
        precomputed_indices = self._precompute_frame_indices()
        
        # Create dataset with precomputed indices
        dataset = MVFoulVideoDataset(
            split_dir=self.split_dir,
            action_dirs=self.action_dirs,
            annotations=self.annotations,
            num_frames=self.extractor.num_frames,
            transform=self.extractor.transform,
            precomputed_frame_indices=precomputed_indices,
            verbose=self.verbose,
        )
        
        # Create dataloader with multiple workers
        num_workers = self.cfg.processing.get("num_workers", 4)
        batch_size = self.cfg.processing.get("batch_size", 1)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.extractor.device != "cpu" else False,
        )
        
        embeddings_dict = {}
        global_idx = 0
        
        desc = f"Processing {self.split} split ({self.cfg.model.variant}, {num_workers} workers)"
        for batch in tqdm(dataloader, desc=desc):
            # Process each sample in batch
            for i in range(len(batch["label"])):
                # Skip failed samples
                if not batch["success"][i]:
                    if not self.cfg.processing.skip_errors:
                        raise ValueError(f"Failed to load video: {batch['video_file'][i]}")
                    continue
                
                # Get video tensor
                video_tensor = batch["video"][i].unsqueeze(0).to(self.extractor.device)  # (1, T, C, H, W)
                
                # Extract embeddings
                with torch.no_grad():
                    embeddings = self.extractor.model(video_tensor)  # (1, T*num_patches, embed_dim)
                    embeddings = embeddings[0].cpu()  # Remove batch dimension
                
                # Pool embeddings
                if self.cfg.embedding.pooling == "mean":
                    pooled_embeddings = embeddings.mean(dim=0)
                elif self.cfg.embedding.pooling == "max":
                    pooled_embeddings = embeddings.max(dim=0)[0]
                elif self.cfg.embedding.pooling == "kmeans":
                    kmeans = KMeans(n_clusters=self.cfg.embedding.kmeans_clusters, random_state=0)
                    kmeans.fit(embeddings.numpy())
                    pooled_embeddings = torch.from_numpy(kmeans.cluster_centers_)
                elif self.cfg.embedding.pooling == "none":
                    pooled_embeddings = embeddings
                else:
                    raise ValueError(f"Unknown pooling: {self.cfg.embedding.pooling}")
                
                # Store in dictionary with global index
                embeddings_dict[global_idx] = {
                    "label": batch["label"][i].item() if isinstance(batch["label"][i], torch.Tensor) else batch["label"][i],
                    "features": pooled_embeddings,
                    "action_class": batch["action_class"][i],
                    "action_dir": batch["action_dir"][i],
                    "video_file": batch["video_file"][i],
                    "angle": batch["angle"][i].item() if isinstance(batch["angle"][i], torch.Tensor) else batch["angle"][i],
                }
                
                global_idx += 1
        
        return embeddings_dict
    
    def _load_frames_as_video(self, action_path: str, frame_files: list) -> Optional[np.ndarray]:
        """Load frames as video array."""
        try:
            frames = []
            # Sample frames uniformly to match model's expected num_frames
            frame_indices = np.linspace(0, len(frame_files) - 1, self.extractor.num_frames, dtype=int)
            
            for idx in frame_indices:
                frame_path = os.path.join(action_path, frame_files[idx])
                import cv2
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Could not load frame {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            return np.stack(frames, axis=0)  # (T, H, W, C)
        except Exception as e:
            print(f"Error loading frames from {action_path}: {e}")
            return None
    
    def save_embeddings(self, embeddings_dict: Dict[int, Dict[str, Any]]):
        """Save embeddings to disk in chunks."""
        dataset_name = self.cfg.dataset.name
        split = self.cfg.dataset.split
        chunk_size = self.cfg.output.get("chunk_size", None)
        
        # Convert dict to list of (idx, data) tuples and sort by index
        sorted_items = sorted(embeddings_dict.items(), key=lambda x: x[0])
        
        # Determine if chunking is needed
        if chunk_size is None or len(sorted_items) <= chunk_size:
            # Save as single file (original behavior)
            output_file = self.output_dir / f"{dataset_name}_{split}_embeddings.pt"
            torch.save(embeddings_dict, output_file)
            if self.verbose:
                print(f"✓ Saved embeddings to {output_file}")
            num_chunks = 1
        else:
            # Save in chunks
            num_chunks = (len(sorted_items) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(sorted_items))
                
                # Build chunk dict preserving original indices
                chunk_dict = {idx: data for idx, data in sorted_items[start_idx:end_idx]}
                
                output_file = self.output_dir / f"{dataset_name}_{split}_embeddings_chunk{chunk_idx:04d}.pt"
                torch.save(chunk_dict, output_file)
                if self.verbose:
                    print(f"✓ Saved chunk {chunk_idx+1}/{num_chunks} ({len(chunk_dict)} samples) to {output_file}")
        
        # Also save config/metadata
        config = {
            "dataset": self.cfg.dataset.name,
            "split": self.cfg.dataset.split,
            "model_variant": self.cfg.model.variant,
            "img_size": self.cfg.model.img_size,
            "num_frames": self.cfg.model.num_frames,
            "pooling": self.cfg.embedding.pooling,
            "num_samples": len(embeddings_dict),
            "classes": {cls.value: cls.name for cls in ActionClass},
            "embed_dim": embeddings_dict[0]["features"].shape[0] if embeddings_dict else None,
            "chunked": chunk_size is not None and len(sorted_items) > chunk_size,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
        }
        
        config_file = self.output_dir / f"{dataset_name}_{split}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        if self.verbose:
            print(f"✓ Saved config to {config_file}")
        
        # Print summary
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Embedding Summary")
            print(f"{'='*60}")
            print(f"  Dataset: {config['dataset']}")
            print(f"  Split: {config['split']}")
            print(f"  Model: {config['model_variant']} ({config['img_size']}x{config['img_size']})")
            print(f"  Total samples: {len(embeddings_dict)}")
            print(f"  Embedding dimension: {config['embed_dim']}")
            print(f"  Pooling: {config['pooling']}")
            print(f"  Classes: {config['classes']}")
            print(f"{'='*60}\n")


@hydra.main(config_path="../conf", config_name="embedding_prepare", version_base=None)
def main(cfg: DictConfig):
    """Main entry point using Hydra configuration."""
    if cfg.output.verbose:
        print("\n" + "="*60)
        print("VJEPA Embedding Preparation")
        print("="*60)
        print(OmegaConf.to_yaml(cfg))
        print("="*60 + "\n")
    
    # Initialize extractor
    extractor = VJEPAEmbeddingExtractor(cfg)
    
    # Initialize preparator
    preparator = MVFoulEmbeddingPreparator(cfg, extractor)
    
    # Prepare embeddings
    embeddings_dict = preparator.prepare()
    
    # Save embeddings
    preparator.save_embeddings(embeddings_dict)
    
    if cfg.output.verbose:
        print("✓ Done!")

if __name__ == "__main__":
    main()

