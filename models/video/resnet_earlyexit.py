import torch.nn as nn
import torchvision.models as models
from .resnet import DistillerBottleneck
from .resnet import DistillerBasicBlock


import sys
sys.path.insert(0, '../../')
from earlyexit.earlyexit_mgr import *


__all__ = ['resnet18_earlyexit', 'resnet50_earlyexit', 'resnet101_earlyexit']

NUM_CLASSES = 2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# def get_input_dim(dim_0):
#     return dim_0 * (dim_1 // 7) ** 2

def get_exits_def_101(ids):
    
    dim_0 = [256, 512, 1024, 2048]
    dim_1 = [56, 28, 14, 7]
    
    layers = [3, 4, 23, 3]

    all_positions = []

    for i in range(4):
        for j in range(layers[i]):
            # print(i, get_input_dim(dim_0[i], dim_1[i]))

            all_positions += [('layer{}.{}.relu3'.format(i+1, j),
                                    nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(dim_0[i], NUM_CLASSES)))]

    if ids[0] == sum(layers) or ids[0] == -1:
        exits_def = all_positions
    else:
        exits_def = [all_positions[i] for i in ids]

    return exits_def

def get_exits_def_50(ids):

    dim_0 = [256, 512, 1024, 2048]
    dim_1 = [56, 28, 14, 7]
    
    layers = [3, 4, 6, 3]

    all_positions = []

    for i in range(4):
        for j in range(layers[i]):
            # print(i, get_input_dim(dim_0[i], dim_1[i]))
           
            all_positions += [('layer{}.{}.relu3'.format(i+1, j),
                                                  nn.Sequential(
                                                  nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(dim_0[i], NUM_CLASSES)))]
            
            # all_positions += [('layer{}.{}.relu3'.format(i+1, j),
            #                         nn.Sequential(
            #                         nn.Conv2d(dim_0[i], 64, kernel_size=3, stride=1, padding=1, bias=True),
            #                         nn.ReLU(),
            #                         nn.AdaptiveAvgPool2d((1, 1)),
            #                         nn.Flatten(),
            #                         nn.Linear(64, NUM_CLASSES)))]
            
            # all_positions += [('layer{}.{}.relu3'.format(i+1, j),
            #                         nn.Sequential(
            #                         nn.Conv2d(dim_0[i], 64, kernel_size=3, stride=1, padding=1, bias=True),
            #                         nn.ReLU(),
            #                         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            #                         nn.ReLU(),
            #                         nn.AdaptiveAvgPool2d((1, 1)),
            #                         nn.Flatten(),
            #                         nn.Linear(64, NUM_CLASSES)))]

    if ids[0] == sum(layers) or ids[0] == -1:
        exits_def = all_positions
    else:
        exits_def = [all_positions[i] for i in ids]

    return exits_def

def get_exits_def_18(ids):
    print(ids)
    dim_0 = [64, 128, 256, 512]
    layers = [2, 2, 2, 2]

    all_positions = []

    for i in range(4):
        for j in range(layers[i]):
            # print(i, get_input_dim(dim_0[i], dim_1[i]))

            all_positions += [('layer{}.{}.relu2'.format(i+1, j),
                                                  nn.Sequential(
                                                  nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(dim_0[i], NUM_CLASSES)))]
            

    if ids[0] == sum(layers) or ids[0] == -1:
        exits_def = all_positions
    else:
        exits_def = [all_positions[i] for i in ids]

    return exits_def


def get_exits_def(arch, ids):
    if arch == "resnet18":
        return get_exits_def_18(ids)
    elif arch == "resnet50":
        return get_exits_def_50(ids)
    elif arch == "resnet101":
        return get_exits_def_101(ids)
    else:
        raise NotImplementedError


class ResNetEarlyExit(models.ResNet):
    def __init__(self, block, layers, ids, arch, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(block, layers, num_classes=num_classes, **kwargs)
        self.ee_mgr = EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def(arch, ids))

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def _resnet(arch, block, layers, ids, pretrained, num_classes=NUM_CLASSES, **kwargs):
    model = ResNetEarlyExit(block, layers, ids, arch, num_classes=NUM_CLASSES, **kwargs)
    # assert not pretrained
    return model

def resnet18_earlyexit(pretrained=False, ids=None, num_classes=NUM_CLASSES, **kwargs):
    """Constructs a ResNet-18 model, with early exit branches.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', DistillerBasicBlock, [2, 2, 2, 2], ids, pretrained, num_classes=NUM_CLASSES,
                   **kwargs)


def resnet50_earlyexit(pretrained=False, ids=None, num_classes=4, **kwargs):
    """Constructs a ResNet-50 model, with early exit branches.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', DistillerBottleneck, [3, 4, 6, 3], ids, pretrained, num_classes=NUM_CLASSES,
                   **kwargs)


def resnet101_earlyexit(pretrained=False, ids=None, num_classes=NUM_CLASSES, **kwargs):
    """Constructs a ResNet-101 model, with early exit branches.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', DistillerBottleneck, [3, 4, 23, 3], ids, pretrained, num_classes=NUM_CLASSES,
                   **kwargs)