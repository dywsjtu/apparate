"""Waymo open dataset dataloader."""
import copy
import csv
import os
import time
from typing import List, Union

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from sklearn.utils import shuffle



DATA_ROOT = os.path.join(os.getenv("HOME"), "waymo/waymo_classification_images")

SAMPLE_LIST_ROOT = os.path.join(DATA_ROOT, "sample_lists", "citywise") 

class WaymoClassification(VisionDataset):
    """
    WaymoClassification dataset.

    Converted from Waymo Open Dataset for classification use.
    """

    def __init__(self, data_root, sample_list_name: Union[str, List[str]],
                 sample_list_root: str = SAMPLE_LIST_ROOT, transform: callable = None, 
                 target_transform: callable = None, resize_res: int = 224,
                 train: bool = False, test: bool = False,
                 label_type: str = 'human', **kwargs):
        """
        Constructor.

        Args
            root (string): Directory containing with the classificaiton images.
            sample_list_name (Either str or list of strs): Name(s) of the
                samplelist csv file(s) to include in this dataset. The list is
                generated using the generate_sample_list method.
            sample_list_root(str, optional): Path to the sample_list csv file.
            If None, root is treated as sample_list_root. Default: None.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            resize_res (int, optional): Image size to resize to.
        """
        super(WaymoClassification, self).__init__(DATA_ROOT)
        self.root = os.path.join(data_root, "waymo_classification_images")
        sample_list_root = os.path.join(self.root, "sample_lists", "citywise")
        self.sample_list_root = sample_list_root
        self.sample_list_name = sample_list_name
        self.sample_list_path = os.path.join(
                self.sample_list_root, f"{self.sample_list_name}.csv")
        self.transform = transform
        self.resize_res = resize_res
        self.label_type = label_type
        self.samples = pd.read_csv(self.sample_list_path)

        if sample_list_name == "pretrain":
            self.samples = shuffle(self.samples, random_state=2023)
            if train: 
                self.samples = self.samples[:int(0.8*len(self.samples))]
            if test:
                self.samples = self.samples[int(0.8*len(self.samples)):]

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete.')
    
        # remove unknown
        self.samples.loc[:, 'class'] -= 1

        if not self.samples[self.samples['class'] > 3].empty:
            raise RuntimeError('class labels not in [0, 3]!')

        # self.update_idxs()

    def update_idxs(self):
        self.samples["idx"] = pd.Series(
            range(0, len(self.samples["idx"]))).values
        self.samples.set_index("idx", inplace=True, drop=False)

    def __getitem__(self, idx):
        """Get a sample.
        Args
            index (int): Index
        Returns
            tuple: (image, target) where target is a tuple of all target types
                if target_type is a list with more than one item.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples.iloc[idx, :]

        img_name = sample['image name']
        if self.label_type == 'human':
            target = sample['class']

        seg_name = sample['segment']
        cam_name = sample['camera name']
        img_path = os.path.join(self.root, seg_name, cam_name, img_name)
        if self.resize_res is not None:
            image = Image.open(img_path).resize(
                (self.resize_res, self.resize_res), resample=PIL.Image.BICUBIC)
        else:
            image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    @property
    def y(self):
        return self.samples["class"].values

    def get_targets(self, class_filter_list=None):
        '''
        Returns
            Targets of elements which belong in the class_filter_list
        '''
        targets = self.samples["class"]
        return targets.values

    def get_class_dist(self):
        """Return class distribution of all samples."""
        class_dist = []
        for i in range(4):
            mask = self.samples['class'] == i
            class_dist.append(len(self.samples[mask]))
        return class_dist