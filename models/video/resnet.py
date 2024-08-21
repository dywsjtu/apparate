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

# The TorchVision implementation in https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# has 2 issues in the implementation of the BasicBlock and Bottleneck modules, which impact our ability to
# collect activation statistics and run quantization:
#   1. Re-used ReLU modules
#   2. Element-wise addition as a direct tensor operation
# Here we provide an implementation of both classes that fixes these issues, and we provide the same API to create
# ResNet and ResNeXt models as in the TorchVision implementation.
# We reuse the original implementation as much as possible.

from collections import OrderedDict
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, _resnet
import os, torch, sys
sys.path.insert(0, '../../')
from earlyexit.modules import EltwiseAdd


__all__ = ['ResNet', 'resnet18', 'resnet50', 'resnet101']#, 'resnet34', 'resnet50', 'resnet101']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}

pretrained_path = os.path.join(os.getenv("HOME"), "urban/urban_pretrain_model")

class DistillerBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(DistillerBasicBlock, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu2'] = self.relu2
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu2(out)

        return out


class DistillerBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(DistillerBottleneck, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        modules['relu2'] = self.relu2
        modules['conv3'] = self.conv3
        modules['bn3'] = self.bn3
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu3'] = self.relu3
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


def resnet18(pretrained=False, ids=None, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = True
    model = ResNet(DistillerBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18'], progress=progress)
        model.load_state_dict(state_dict)
    return model


# def resnet34(pretrained=False, ids=None, progress=True, **kwargs):
#     """Constructs a ResNet-34 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """

#     model = ResNet(DistillerBasicBlock, [3, 4, 6, 3], num_classes = 6, **kwargs)
#     return model


def resnet50(pretrained=False, ids=None, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = True
    model = ResNet(DistillerBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'], progress=progress)
        # print(state_dict)
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained=False, ids=None, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = True
    model = ResNet(DistillerBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101'], progress=progress)
        # print(state_dict)
        model.load_state_dict(state_dict)
    return model
