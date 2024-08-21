#
# Copyright (c) 2019 Intel Corporation
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

"""Resnet for CIFAR10 with Early Exit branches

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
import torch
from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar
import torch.nn as nn
import sys
sys.path.insert(0, '../../')
from earlyexit.earlyexit_mgr import *


__all__ = ['resnet20_cifar10_earlyexit', 'resnet32_cifar10_earlyexit', 'resnet44_cifar10_earlyexit',
           'resnet56_cifar10_earlyexit', 'resnet110_cifar10_earlyexit', 'resnet1202_cifar10_earlyexit']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def get_exits_def(ids, layers=[9, 9, 9]):

    dims = [1600, 800, 256]

    all_positions = []

    for i in range(3):
        for j in range(layers[i]):
            all_positions += [('layer{}.{}.relu2'.format(i+1, j), nn.Sequential(nn.AvgPool2d(3),
                                                nn.Flatten(),
                                                nn.Linear(dims[i], NUM_CLASSES)))]
    print(ids)
    if ids[0] == sum(layers):
        exits_def = all_positions
    else:
        exits_def = [all_positions[i] for i in ids]
                        
    return exits_def


class ResNetCifarEarlyExit(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def(args[-1], args[-2]))

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def resnet20_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [3, 3, 3], ids, **kwargs)
    return model

def resnet32_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [5, 5, 5], ids, **kwargs)
    return model

def resnet44_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [7, 7, 7], ids, **kwargs)
    return model

def resnet56_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [9, 9, 9], ids, **kwargs)
    return model

def resnet110_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [18, 18, 18], ids, **kwargs)
    return model

def resnet1202_cifar10_earlyexit(ids, **kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [200, 200, 200], ids, **kwargs)
    return model