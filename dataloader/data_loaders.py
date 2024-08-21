#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Helper code for data loading.
"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.sampler import Sampler
from functools import partial
import numpy as np
from .urban_dataset import UrbanClassification
from .waymo_dataset import WaymoClassification
from .video_dataset import VideoClassification

import utils

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


def classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10' 
    elif 'waymo' in arch:
        dataset = 'waymo'
    elif 'urban' in arch:
        dataset = 'urban'
    else:
        dataset = 'video'
    return dataset


def __dataset_factory(dataset, arch=None):
    dataset_factory_dict = {
        'cifar100': cifar100_get_datasets,
        'cifar10': cifar10_get_datasets,
        'waymo' : waymo_get_datasets,
        'urban' : urban_get_datasets,
        'video': video_get_datasets,       
    }

    # set up glue datasets for nlp workloads
    for glue_dataset in utils.all_nlp_datasets:
        dataset_factory_dict[glue_dataset] = glue_get_datasets

    return dataset_factory_dict.get(dataset, None)


def load_data(dataset, data_dir,
              batch_size, arch=None, workers=1, test_only=False, tokenizer=None, ):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the dataset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        tokenizer (transformer.tokenizer): tokenizer for language models. 
            Returns None for other workloads.
        arch: model name (resnet18_auburn)
    """
    # if dataset not in utils.all_supported_datasets:
    #     raise ValueError(f"load_data does not support dataset {dataset}")
    datasets_fn = __dataset_factory(dataset, arch)
    return get_data_loaders(datasets_fn, data_dir, dataset, arch, batch_size, workers, test_only=test_only, tokenizer=tokenizer)


def cifar10_get_datasets(data_dir, dataset, load_train=True, load_test=True):
    """Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.201])
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.201])
        ])

        test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir, dataset, load_train=True, load_test=True):
    """Load the CIFAR100 dataset.
    """
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.507, 0.4865, 0.4409], std = [0.2673, 0.2564, 0.2761])
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                         download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.507, 0.4865, 0.4409], std = [0.2673, 0.2564, 0.2761])
        ])

        test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                        download=True, transform=test_transform)

    return train_dataset, test_dataset


def waymo_get_datasets(data_dir, dataset, load_train=True, load_test=True):
    """Load the waymo dataset.
    """
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        train_dataset = WaymoClassification(data_dir, sample_list_name="waymo", train=True, transform=train_transform)

    test_dataset = None
    if load_test:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        test_dataset = WaymoClassification(data_dir, sample_list_name="waymo", test=True, transform=train_transform)

    return train_dataset, test_dataset

def urban_get_datasets(data_dir,  dataset, load_train=True, load_test=True):
    """Load the urban dataset.
    """
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        train_dataset = UrbanClassification(data_dir, sample_list_name="urban", train=True, transform=train_transform)

    test_dataset = None
    if load_test:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        test_dataset = UrbanClassification(data_dir, sample_list_name="urban", test=True, transform=train_transform)

    return train_dataset, test_dataset


def video_get_datasets(data_dir, arch, load_train, load_test):
    """Load the urban dataset.
    """
    train_dataset = None

    print(data_dir, arch, load_train, load_test)

    if load_train:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        train_dataset = VideoClassification(arch, sample_list_name=data_dir, train=True, transform=train_transform)

    test_dataset = None
    if load_test:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        test_dataset = VideoClassification(arch, sample_list_name=data_dir, test=True, transform=train_transform)

    return train_dataset, test_dataset
    
def glue_get_datasets(data_dir, dataset, tokenizer, load_train=True, load_test=True):
    train_dataset, test_dataset = None, None
    if load_train:
        train_dataset, _ = load_and_cache_examples(data_dir, dataset, tokenizer, evaluate=False)
    if load_test:
        test_dataset, _ = load_and_cache_examples(data_dir, dataset, tokenizer, evaluate=True)
    return train_dataset, test_dataset

def load_and_cache_examples(data_dir, dataset, tokenizer, evaluate=False, streaming_example_ids=[]):
    overwrite_cache = False  # don't overwrite the cached training and evaluation sets
    # The maximum total input sequence length after tokenization.
    # sequences longer than this will be truncated, sequences shorter will be padded.
    max_seq_length = 128
    data_dir = os.path.join(data_dir, dataset.upper())  # add dataset to path
    processor = processors[dataset]()
    output_mode = output_modes[dataset]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            "two_stage",
            str(max_seq_length),
            str(dataset),
        ),
    )

    if os.path.exists(cached_features_file) and not overwrite_cache:
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print(f"Creating features from dataset file at {data_dir}")
        label_list = processor.get_labels()
        examples = (
            processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
        )
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    if streaming_example_ids != []:
        # convert raw dataset to streaming based on given array of example ordering
        new_features = []
        for sample_id in streaming_example_ids:
            new_features.append(features[sample_id])
        features = new_features
    
     # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.long)

    if features[0].token_type_ids is None:
        # For RoBERTa (a potential bug!)
        all_token_type_ids = torch.tensor([[0] * max_seq_length for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.long)
        print(f"For RoBERTa (a potential bug!)")
    else:
        all_token_type_ids = torch.tensor([f.token_type_ids for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for i, f in enumerate(features) if i >= len(features) / 10], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
    # load orig model output?

    return dataset, len(all_labels)

def get_data_loaders(datasets_fn, data_dir, dataset, arch, batch_size, num_workers, test_only=False, tokenizer=None):
    """Get train and test data loaders for a given dataset.
    """
    if tokenizer is None:
        if dataset == 'video':  
            train_dataset, test_dataset = datasets_fn(data_dir, arch, True, True)
        else:
            train_dataset, test_dataset = datasets_fn(data_dir, dataset, True, True)
        
    else:
        train_dataset, test_dataset = datasets_fn(data_dir, dataset, tokenizer, load_train=True, load_test=True)

    # eval_sampler = SequentialSampler(test_dataset)  # potential pitfall: deebert uses this in evaluate()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers, 
                                              pin_memory=True)

        
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=True)
    if test_only:
        return None, test_loader
    return train_loader, test_loader
