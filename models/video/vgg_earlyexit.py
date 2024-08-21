import os, sys
import torch.nn as nn
import torchvision.models as nnmodels
from .vgg import make_layers, cfg, VGG, vgg11, vgg13
sys.path.insert(0, '../../')
from earlyexit.earlyexit_mgr import *

from summary_graph import *

__all__ = ['vgg11_earlyexit', 'vgg13_earlyexit', 'vgg16_earlyexit']

NUM_CLASSES = 2

class VGGEarlyExit(VGG):
    def __init__(self, features, ids, arch, **kwargs):
        super().__init__(features, **kwargs)
    
        self.ee_mgr = EarlyExitMgr()
        model_profile_path = os.path.join(os.getenv("HOME"), 'apparate', \
                                           'profile_pickles_bs', '{}_profile.pickle'.format(arch))
        print(model_profile_path, ids)
        self.ee_mgr.attach_exits(self, get_exits_def(self, arch, ids, model_profile_path))
        
    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def vgg11_earlyexit(pretrained=False, ids=None, num_classes=NUM_CLASSES, **kwargs):
    model = VGGEarlyExit(make_layers(cfg['A']), ids, 'vgg11', **kwargs)
    return model

def vgg13_earlyexit(pretrained=False, ids=None, num_classes=NUM_CLASSES, **kwargs):
    model = VGGEarlyExit(make_layers(cfg['B']), ids, 'vgg13', **kwargs)
    return model

def vgg16_earlyexit(pretrained=False, ids=None, num_classes=NUM_CLASSES, **kwargs):
    model = VGGEarlyExit(make_layers(cfg['D']), ids, 'vgg16', **kwargs)
    return model

