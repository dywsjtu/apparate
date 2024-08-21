"""Wrapper of Mp4Video video."""
import csv
import copy
import os
import glob
import subprocess
from hashlib import md5
from typing import List, Union

import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from sklearn.utils import shuffle

DATA_ROOT = os.path.join(os.getenv("HOME"), "urban")

SAMPLE_LIST_ROOT = os.path.join(DATA_ROOT, "sample_lists", "citywise") 

class UrbanClassification(VisionDataset):
    def __init__(self, data_root, sample_list_name: Union[str, List[str]],
                 sample_list_root: str = SAMPLE_LIST_ROOT, transform: callable = None, 
                 target_transform: callable = None, resize_res: int = 224,
                 train: bool = False, test: bool = False,
                 label_type: str = 'human', **kwargs):


        super(UrbanClassification, self).__init__(DATA_ROOT)
        self.root = data_root
        sample_list_root = os.path.join(data_root, "sample_lists", "citywise")
        use_cache = True
        self.resize_res = resize_res
        self.use_cache = use_cache
        self.sample_list_name = sample_list_name
        self.sample_list_root = sample_list_root
        if isinstance(self.sample_list_name, list):
            self.sample_list_path = [os.path.join(
                self.sample_list_root, f"{s}.csv") for s in
                self.sample_list_name]
        else:
            self.sample_list_path = os.path.join(
                self.sample_list_root, f"{self.sample_list_name}.csv")

        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.label_type = label_type

        if isinstance(self.sample_list_path, list):
            self.samples = pd.concat([pd.read_csv(p)
                                      for p in self.sample_list_path], axis=0)
        else:
            self.samples = pd.read_csv(self.sample_list_path)

        self.samples['class'] = self.samples['class'].astype(int)
        self.samples['frame id'] = self.samples['frame id'].astype(int)
        self.update_idxs()

        if sample_list_name == "pretrain":
            self.samples = shuffle(self.samples, random_state=2023)
            if train: 
                self.samples = self.samples[:int(0.8*len(self.samples))]
            if test:
                self.samples = self.samples[int(0.8*len(self.samples)):]


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
        
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sample = self.samples.iloc[idx, :]
        image = self.get_image(idx)
        
        if self.label_type == 'human':
            target = sample['class']
        elif self.label_type == 'golden_label':
            target = sample['golden_label']
        else:
            raise RuntimeError(f"Unrecognized label_type: {self.label_type}")
        if self.target_transform:
            image = self.target_transform(target)
        return image, target

    def get_image(self, index):
        '''Read a frame from disk and crop out an object for classification.

        Args
            index(int)

        Return cropped image
        '''
        sample = self.samples.iloc[index, :]
        frame_idx = int(sample['frame id'])
        img_path = os.path.join(self.root, sample['camera'], 'frame_images',
                                "{:06d}.jpg".format(frame_idx))

        cache_file_path = os.path.join(
            self.root, sample['camera'], 'classification_images',
            '{:06d}_{}_{}_{}_{}.jpg'.format(
                int(sample['frame id']), int(sample['xmin']),
                int(sample['ymin']), int(sample['xmax']), int(sample['ymax'])))
        if self.use_cache and os.path.exists(cache_file_path):
            # Cache hit
            cropped = Image.open(cache_file_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
            cropped = image.crop((sample["xmin"], sample["ymin"],
                                  sample["xmax"], sample["ymax"])).resize(
                (self.resize_res, self.resize_res), resample=PIL.Image.BICUBIC)
            cropped.save(cache_file_path)
        if self.transform:
            cropped = self.transform(cropped)
        return cropped

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    def get_md5(self):
        return md5(','.join(
            (str(self.samples['frame id'].values))).encode('utf-8')).hexdigest()

    @property
    def y(self):
        return self.samples["class"].values


# C = UrbanClassification('pretrain')
# print(C[0])