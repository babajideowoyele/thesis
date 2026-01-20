#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MVFoul dataset. """

import json
import os

import numpy as np
import cv2
from torchvision.transforms import Compose, v2
import torchvision.transforms as transforms
import torch
from datasets.utils.video.randerase import RandomErasing
from datasets.utils.video.transforms import RandomResizedCropAndInterpolation
from utils.logging import get_logger
from datasets.utils.mvfoul_translations import translate_annotation, ActionClass

logger = get_logger(__name__)


class Mvfoul(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(Mvfoul, self).__init__() 
        self.cfg = cfg
        self.split = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self.take_frames = self.get_num_frames(cfg)
        self.overfit: bool = cfg.DATA.OVERFIT.ENABLE
        self.num_overfit_samples: int = cfg.DATA.OVERFIT.NUM_SAMPLES
        self.weights = None
        self._construct_dataset()
        self._config_transform()

    @staticmethod
    def get_num_frames(cfg):
        assert cfg.DATA.NUM_INPUT_FRAMES is not None, "NUM_INPUT_FRAMES must be specified."
        
        if cfg.DATA.TAKE_NUM_FRAMES is not None:
            tf = cfg.DATA.TAKE_NUM_FRAMES
            assert tf >= cfg.DATA.NUM_INPUT_FRAMES, "TAKE_NUM_FRAMES must be greater than or equal to NUM_INPUT_FRAMES."
        else:
            tf = cfg.DATA.NUM_INPUT_FRAMES
        assert tf % cfg.DATA.NUM_INPUT_FRAMES == 0, "TAKE_NUM_FRAMES must be divisible by NUM_INPUT_FRAMES."

        return tf
    
    def _construct_dataset(self):
        if self.split == "train":
            self.crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            self.crop_size = self.cfg.DATA.TEST_CROP_SIZE


        path = os.path.join(
            self.data_root_dir,
            self.split)
        assert os.path.exists(path), "{} does not exist".format(path)
        self.data_root_dir = path

        entries_unordered = os.listdir(self.data_root_dir)

        entries: list[str] = ["action_" + str(i) for i in range(len(entries_unordered)) if "action_" + str(i) in entries_unordered]

        with open(os.path.join(self.data_root_dir, "annotations.json"), "r") as f:
            annotations = json.load(f)
            self.num_annotations = annotations["Number of actions"] or len(annotations["Actions"])
            self.annotations = annotations["Actions"]
        
        self.labels = [None] * self.num_annotations
        
        self._process_labels(self.annotations)


        if self.overfit:
            list_classes = {cls: 0 for cls in ActionClass.get_classes()}
            self.overfit_dirs = []
            self.overfit_labels = []
            for idx, label in enumerate(self.labels):
                action_class = label["type"]
                action_enum = ActionClass(action_class)
                if list_classes[action_enum] < self.num_overfit_samples:
                    self.overfit_dirs.append(entries[idx])
                    self.overfit_labels.append(label)
                    list_classes[action_enum] += 1


        self.dirs = [e for e in entries if os.path.isdir(os.path.join(self.data_root_dir, e))]
        
        self.meta_data = {e: len(os.listdir(os.path.join(self.data_root_dir, e))) for e in self.dirs}

    def __len__(self):
        if not self.overfit:
            if self.split == "train" and self.cfg.DATA.UNDERSAMPLE.ENABLE:
                return sum([1 for w in self.get_weights() if w > 0])
            return len(self.labels)
        else:
            return len(self.overfit_labels)

    def __getitem__(self, index):
        if not self.overfit:
            dir_name = self.dirs[index]
            label = self.labels[index]
        else:
            idx = index % len(self.overfit_labels)
            dir_name = self.overfit_dirs[idx]
            label = self.overfit_labels[idx]
        assert label == self.labels[int(dir_name.split("_")[1])]
        feature = self._read_videos_from_dir(os.path.join(self.data_root_dir, dir_name))
        return feature[0], feature[1], {'supervised': label,
                                        'meta_data': {"dir_name": dir_name},}
    
    def get_weights(self):
        if self.weights is None:
            class_counts = {}
            labels = self.labels if not self.overfit else self.overfit_labels
            for label in labels:
                action_class = label["type"]
                if action_class not in class_counts:
                    class_counts[action_class] = 0
                class_counts[action_class] += 1
            if self.cfg.TRAIN.UNDERSAMPLE.ENABLE and self.split == "train":
                min_count = min(class_counts.values())
                for action_class in class_counts:
                    class_counts[action_class] = min(class_counts[action_class], min_count * self.cfg.TRAIN.UNDERSAMPLE.RATE)
            
            factors = {action_class: 1.0 / count for action_class, count in class_counts.items()}
            
            weights: list[float] = []
            for label in labels:
                if class_counts[label["type"]] <= 0:
                    weights.append(0.0)
                    continue
                action_class = label["type"]
                weight = float(factors[action_class])
                weights.append(weight)
                class_counts[action_class] -= 1
            self.weights = weights
        
        return self.weights


    def _read_videos_from_dir(self, dir_path):
        video_files = sorted(os.listdir(dir_path))
        video_tensors = []
        for vf in video_files:
            video_path = os.path.join(dir_path, vf)
            video_tensor = self._load_video(video_path)
            video_tensors.append(video_tensor)
            if len(video_tensors) == self.cfg.DATA.NUM_VIEWS:
                break
        actual_views = len(video_tensors)
        if actual_views < self.cfg.DATA.NUM_VIEWS:
            # pad with empty tensors
            num_missing = self.cfg.DATA.NUM_VIEWS - actual_views
            shape = video_tensors[0].shape
            for _ in range(num_missing):
                video_tensors.append(torch.zeros(shape))
        video_tensor = torch.cat(video_tensors, dim=0)  #(num_views, C, T, H, W)

        mask = torch.full((self.cfg.DATA.NUM_VIEWS,), 2, dtype=torch.long)
        mask[0] = 1
        mask[actual_views:] = 0

        return video_tensor, mask
    
    def _load_video(self, video_path):
        vid = cv2.VideoCapture(video_path)

        count, success = 0, True
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        height, width = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

        indices = self._custom_sampling(
            vid_length=num_frames,
            vid_fps=fps,
            num_frames=self.take_frames,
            interval=2,
            height=height,
            width=width,
        )
        frames = []
        for idx in indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
            success, frame = vid.read()
            if success:
                frames.append(frame)


        vid.release()
        selected: torch.Tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
        
        if not self.cfg.PRETRAIN.ENABLE:
            selected = self.transform(selected.permute(0, 3, 1, 2))  #  C, T, H, W
        else:
            selected = selected.permute(3,0,1,2)
        return selected.unsqueeze(0)


    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "path": indicating the target's path w.r.t. index
                "supervised_label": indicating the class of the target 
        """
        sample_info = {}
        return sample_info

    def _custom_sampling(self, vid_length,
            vid_fps,
            num_frames,
            interval,
            height,
            width):
        if self.cfg.DATA.CENTER_FRAME > vid_length:
            center_frame = vid_length // 2
        else:
            center_frame = self.cfg.DATA.CENTER_FRAME
        
        if center_frame + self.cfg.DATA.SAMPLING_RATE * (num_frames // 2) > vid_length:
            final_frame = vid_length - 1
        else:
            final_frame = center_frame + self.cfg.DATA.SAMPLING_RATE * (num_frames // 2) 
        if center_frame - self.cfg.DATA.SAMPLING_RATE * (num_frames // 2) < 0:
            start_frame = 0
        else:
            start_frame = center_frame - self.cfg.DATA.SAMPLING_RATE * (num_frames // 2)
        indices = torch.linspace(start_frame, final_frame, steps=num_frames).long()
        return indices
    
    def _process_labels(self, annotations):
        for idx, annotation in annotations.items():
            label = translate_annotation(annotation)
            self.labels[int(idx)] = label
    

    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None
        fill_value = [int(x * 255) for x in self.cfg.DATA.MEAN]
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                v2.ToImage(),                          # 1. Convert to tensor subclass
                v2.RandAugment(
                    num_ops=2, 
                    magnitude=9,
                    fill=fill_value, # This replaces the 'mean' functionality in timm
                    interpolation=v2.InterpolationMode.BILINEAR 
                ) if self.cfg.AUGMENTATION.AUTOAUGMENT.ENABLE else v2.Identity(),
                v2.ToDtype(torch.float32, scale=True),
                transforms.RandomHorizontalFlip()
            ]
            
            if self.cfg.DATA.TRAIN_JITTER_SCALES[0] <= 1:
                std_transform_list += [transforms.RandomResizedCrop(
                        size=self.cfg.DATA.TRAIN_CROP_SIZE,
                        scale=[
                            self.cfg.DATA.TRAIN_JITTER_SCALES[0],
                            self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                        ],
                        ratio=self.cfg.AUGMENTATION.RATIO
                    ),]
            else:
                std_transform_list += [RandomResizedCropAndInterpolation(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    scale=self.cfg.DATA.TRAIN_JITTER_SCALES,
                    ),]

            if self.cfg.AUGMENTATION.COLOR_AUG:
                color_jitter = v2.ColorJitter(
                            brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                            contrast=self.cfg.AUGMENTATION.CONTRAST,
                            saturation=self.cfg.AUGMENTATION.SATURATION,
                            hue=self.cfg.AUGMENTATION.HUE)
                std_transform_list += [
                    v2.RandomApply([color_jitter], p=self.cfg.AUGMENTATION.COLOR_P),
                    v2.RandomGrayscale(p=self.cfg.AUGMENTATION.GRAYSCALE),
                ]
            std_transform_list += [
                v2.Normalize(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=False
                ),
                RandomErasing(self.cfg)
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'val' or self.split == 'test':
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                v2.ToImage(),                          # 1. Convert to tensor subclass
                v2.ToDtype(torch.float32, scale=True),
                self.resize_video,
                v2.Normalize(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=False
                )
            ]
            self.transform = Compose(std_transform_list)
