#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MVFoul dataset. """

import json
import os
import cv2
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch
from tadaconv.datasets.utils.random_erasing import RandomErasing
from tadaconv.datasets.utils.transformations import ColorJitter, KineticsResizedCrop
import tadaconv.utils.logging as logging
from tadaconv.datasets.base.builder import DATASET_REGISTRY
from tadaconv.utils.mvfoul_translation import translate_annotation, ActionClass

logger = logging.get_logger(__name__)



@DATASET_REGISTRY.register()
class Mvfoul(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(Mvfoul, self).__init__() 
        self.cfg = cfg
        self.split = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self.take_frames = self.get_num_frames(cfg)
        self.overfit: bool = cfg.DATA.OVERFIT.ENABLE
        self.num_overfit_samples: int = cfg.DATA.OVERFIT.NUM_SAMPLES
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
        return len(self.labels) if not self.overfit else len(self.overfit_labels) 

    def __getitem__(self, index):
        if not self.overfit:
            dir_name = self.dirs[index]
            label = self.labels[index]
        else:
            idx = index % len(self.overfit_labels)
            dir_name = self.overfit_dirs[idx]
            label = self.overfit_labels[idx]
        feature = self._read_videos_from_dir(os.path.join(self.data_root_dir, dir_name))
        return feature[0], feature[1], {'supervised': label,
                                        'meta_data': {"dir_name": dir_name},}


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
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            interval=2,
            height=height,
            width=width,
        )
        indices = torch.linspace(0, num_frames - 1, steps=self.take_frames).tolist()
        frames = []
        for idx in indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = vid.read()
            if success:
                frames.append(frame)


        vid.release()
        selected = torch.stack([torch.from_numpy(frame) for frame in frames])
        
        if not self.cfg.PRETRAIN.ENABLE:
            selected = self.transform(selected)  #  C, T, H, W
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
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo()
            ]
            
            if self.cfg.DATA.TRAIN_JITTER_SCALES[0] <= 1:
                std_transform_list += [transforms.RandomResizedCropVideo(
                        size=self.cfg.DATA.TRAIN_CROP_SIZE,
                        scale=[
                            self.cfg.DATA.TRAIN_JITTER_SCALES[0],
                            self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                        ],
                        ratio=self.cfg.AUGMENTATION.RATIO
                    ),]
            else:
                std_transform_list += [KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]
            if self.cfg.AUGMENTATION.AUTOAUGMENT.ENABLE:
                from tadaconv.datasets.utils.auto_augment import creat_auto_augmentation
                std_transform_list.append(creat_auto_augmentation(self.cfg.AUGMENTATION.AUTOAUGMENT.TYPE, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.MEAN))
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        color=self.cfg.AUGMENTATION.COLOR_P,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        ),
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
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
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)
