# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator


def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]
    joints_generator = [
        JointsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_JOINTS,
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        joints_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)
    
    print("Total number of training images:", len(dataset))

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        transforms
    )

    print("Total number of test images:", len(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
    
    
#################################################################
##### Single Image Test #####
################################################################

import cv2
import torch
from torch.utils.data import Dataset

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transforms=None, dataset_name='COCO'):
        self.image_path = image_path
        self.transforms = transforms
        self.name = dataset_name

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Read the image using OpenCV (BGR format)
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError("Could not load image at {}".format(self.image_path))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # If a transform is provided, apply it
        if self.transforms:
            img = self.transforms(img)
        return img, {} # For compatibility, return an empty dict for annotations


def make_single_test_dataloader(cfg, image_path):
    transforms = None
    dataset = SingleImageDataset(image_path, transforms=transforms, dataset_name='COCO')
    print("Total number of test images in single-image dataset:", len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    return data_loader, dataset


###############################################################
###### Live Stream #############
###############################################################

import cv2
import torch
import torchvision
from torch.utils.data import IterableDataset

class LiveStreamIterableDataset(IterableDataset):
    def __init__(self, gst_pipeline, transforms=None):
        self.gst_pipeline = gst_pipeline # gst_pipeline: a full GStreamer launch string
        self.transforms   = transforms

    def _open_capture(self):
        cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.gst_pipeline}")
        return cap

    def __iter__(self):
        # If DataLoader:num_workers > 1, each worker runs this separately
        cap = self._open_capture()
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    cap.release()
                    cap = self._open_capture()
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB NumPy and yield
                yield frame_rgb, {}
        finally:
            cap.release()


def make_live_stream_dataloader(cfg, gst_pipeline):
    transforms = None
    ds = LiveStreamIterableDataset(gst_pipeline, transforms)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    return loader, ds

