from summary_graph import get_exits_def
import utils
import os
import sys
import torch
import torch.nn as nn
from .earlyexit_mgr import EarlyExitMgr
from .earlyexit_mgr import _find_module, _split_module_name, _get_branch_point_module
sys.path.insert(1, '../')
sys.path.insert(1, '../..')


class EarlyExitModel(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model  # EE-capable model with all ramps trained and attached
        self.device = None  # target device for the model
        self.ee_mgr = EarlyExitMgr()

    @property
    def num_layers(self):
        if hasattr(self.model, "num_layers"):
            return self.model.num_layers
        else:
            return None
    # # XXX(ruipan): reimplement .to(), .eval()...? or just inhereit from nn.Module?
    # def to(self, device):
    #     self.model.to(device)
    # def eval(self):
    #     self.model.eval()

    def forward(self, *args, **kwargs):
        # print(f"forward is triggered, device: {next(self.model.parameters()).device}")
        self.ee_mgr.delete_exits_outputs(self.model)
        # Run the input through the network (including exits)
        x = self.model.forward(*args, **kwargs)
        ee_outputs = self.ee_mgr.get_exits_outputs(self.model)
        return x, ee_outputs
    
    def activate_ramps(self, ramp_ids, all_exit_def, shadow_ramp_idx=None):
        """
        Activate ramps corresponding to ramp_ids.
        
        Args:
            ramp_ids (list of int): list of ramp ids to inject and activate.
            all_exit_def (list of (module_name, torch.nn.Module)): list of ramps with weights loaded
        """
        exit_def = [all_exit_def[i] for i in ramp_ids]
        self.ee_mgr.attach_exits(self.model, exit_def, shadow_ramp_idx=shadow_ramp_idx)

    def deactivate_ramp(self, ramp_id, all_exit_def):
        """
        Deactivate ramp corresponding to ramp_id.

        Args:
            ramp_ids (list of int): list of ramp ids to inject and activate.
            all_exit_def (list of (module_name, torch.nn.Module)): list of ramps with weights loaded
        """
        node_name = all_exit_def[ramp_id][0]
        self.ee_mgr.detach_exit(self.model, node_name)