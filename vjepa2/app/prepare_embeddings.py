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

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_large_rope, vit_giant_xformers_rope
from src.datasets.utils.mvfoul_translations import ActionClass




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
        self.IMAGENET_DEFAULT_MEAN = cfg.data.IMAGENET_DEFAULT_MEAN
        self.IMAGENET_DEFAULT_STD = cfg.data.IMAGENET_DEFAULT_STD
        
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
            video = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
            
            return video
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    @torch.no_grad()
    def extract_embeddings(self, video: np.ndarray) -> torch.Tensor:
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
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        
        # Apply transform
        video_tensor = self.transform(video_tensor)
        
        # Add batch dimension and move to device
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        embeddings = self.model(video_tensor)  # (B, T*num_patches, embed_dim)
        
        return embeddings[0].cpu()  # Remove batch dimension and return to CPU


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
        self.action_dirs = sorted([e for e in entries_unordered if e.startswith("action_")])
    
    def prepare(self) -> Dict[int, Dict[str, Any]]:
        """
        Prepare embeddings for all videos in the dataset.
        
        Returns:
            Dictionary mapping index to {label, features}
        """
        embeddings_dict = {}
        
        desc = f"Processing {self.split} split ({self.cfg.model.variant})"
        for idx, action_dir in enumerate(tqdm(self.action_dirs, desc=desc)):
            try:
                # Get label
                annotation = self.annotations[idx]
                label = annotation["type"]
                
                # Get videos from the action directory
                action_path = os.path.join(self.split_dir, action_dir)
                video_files = sorted([f for f in os.listdir(action_path) if f.endswith(".mp4")])
                
                if not video_files:
                    if self.verbose:
                        print(f"⚠ No videos found in {action_path}")
                    if not self.cfg.processing.skip_errors:
                        raise ValueError(f"No videos in {action_path}")
                    continue
                
                # Load videos as video
                video = self._load_videos_as_video(action_path, video_files)
                
                if video is None:
                    if not self.cfg.processing.skip_errors:
                        raise ValueError(f"Failed to load video from {action_path}")
                    continue
                
                # Extract embeddings
                embeddings = self.extractor.extract_embeddings(video)
                
                if embeddings is None:
                    if not self.cfg.processing.skip_errors:
                        raise ValueError(f"Failed to extract embeddings")
                    continue
                
                # Pool embeddings
                if self.cfg.embedding.pooling == "mean":
                    pooled_embeddings = embeddings.mean(dim=0)
                elif self.cfg.embedding.pooling == "max":
                    pooled_embeddings = embeddings.max(dim=0)[0]
                elif self.cfg.embedding.pooling == "none":
                    pooled_embeddings = embeddings
                else:
                    raise ValueError(f"Unknown pooling: {self.cfg.embedding.pooling}")
                
                # Store in dictionary
                embeddings_dict[idx] = {
                    "label": label,
                    "features": pooled_embeddings,
                    "action_class": ActionClass(label).name,
                }
                
                if self.verbose and (idx + 1) % self.cfg.processing.log_interval == 0:
                    print(f"  Processed {idx + 1}/{len(self.action_dirs)} videos")
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Error processing {action_dir} (idx={idx}): {e}")
                if not self.cfg.processing.skip_errors:
                    raise
                continue
        
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
        """Save embeddings to disk."""
        dataset_name = self.cfg.dataset.name
        split = self.cfg.dataset.split
        
        # Save as checkpoint-style file
        output_file = self.output_dir / f"{dataset_name}_{split}_embeddings.pt"
        torch.save(embeddings_dict, output_file)
        if self.verbose:
            print(f"✓ Saved embeddings to {output_file}")
        
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

