# modified from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import OrderedDict
import numpy as np
try:
    from deepspeed.accelerator import get_accelerator
    deepspeed_imported = True
except ImportError:
    deepspeed_imported = False
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"Successfully imported profiler without deepspeed as cuda is available")
    else:
        print(f"Warning: torch.cuda unavailable, profiling on CPUs!")
        

Tensor = torch.Tensor

module_flop_count = []
module_mac_count = []
old_functions = {}

# NOTE(ruipan): copied from https://github.com/TylerYep/torchinfo/blob/main/torchinfo/layer_info.py
DETECTED_INPUT_OUTPUT_TYPES = Union[
    Sequence[Any], Dict[Any, torch.Tensor], torch.Tensor
]


class Profile(object):
    def __init__(
        self,
        name: str,
        full_name: str,
        type: str,
        depth: int,
        num_params: int,
        input_shape: list,
        output_shape: list,
        input_elem_bytes: int,
        output_elem_bytes: int,
        fwd_latency: float,
        fwd_latency_pctg: float,
        macs: float,
        fwd_flops: float,
        children: list = [],
        vanilla_latency_up_until_me: float = 0.0,
        ramp_latencies_up_until_me: list = [],
    ):  # each Profile corresponds to the profile of a torch.nn module
        self.name = name  # torch.nn module name
        self.full_name = full_name  # e.g., layer1.0.conv1
        self.type = type  # torch.nn module type
        self.depth = depth  # depth of current module in model
        self.num_params = num_params  # number of parameters
        self.num_params_pctg = None  # percentage of total params
        # shape of input/output tensor
        self.input_shape = input_shape
        self.output_shape = output_shape
        # size in bytes of an individual element in the input/output tensor
        self.input_elem_bytes = input_elem_bytes
        self.output_elem_bytes = output_elem_bytes
        # fwd latency (forward propagation latency) in ms
        self.fwd_latency = fwd_latency
        # fwd latency obtained by profiler, which might have inconsistencies
        self.fwd_latency_orig = fwd_latency
        self.fwd_latency_pctg = fwd_latency_pctg  # percentage of total fwd latency
        # latency in ms after data passes through me
        self.vanilla_latency_up_until_me = vanilla_latency_up_until_me
        # list of latencies of ramps before current branchpoint
        self.ramp_latencies_up_until_me = ramp_latencies_up_until_me
        # NOTE(ruipan): the following aren't matching with the
        # original deepspeed profiler's output... fix later if needed
        self.macs = macs  # number of multiply-accumulate operations (MACs)
        self.macs_pctg = None  # percentage of total MACs
        # number of floating-point operations (flops) OR floating-point operations per second (FLOPS)??
        self.fwd_flops = fwd_flops
        self.fwd_flops_pctg = None
        self.children = children

    def set_child_modules(self, children) -> None:
        """Sets up the child modules, and fills in 
        the overall percentage statistics

        Args:
            children (Profile): profile of child of module
            of which the current profile is from
        """
        self.children = children
        if self.name == "model":  # outermost model
            self.fwd_latency = self.calibrate_latency()  # make each parent's latency the sum of its childrens' latencies
            
            # after calibrate_latency, for outermost model, we could have: 
            # fwd_latency_orig 12ms and fwd_latency 8ms, but the wall clock time is actually 12ms.
            # thus, we do normalize_latency to eliminate this discepancy by increasing all modules'
            # latencies by a fixed factor.
            normalization_factor = self.fwd_latency_orig / self.fwd_latency
            self.normalize_latency(normalization_factor)            
            
            # fills in overall percentage statistics and layer full names
            total_duration = self.fwd_latency
            self.calculate_overall_stats(
                total_duration=total_duration, parents_name="model")
            # the following can be used on all models whose modules are sequentially traversed
            self.vanilla_latency_up_until_me = self.calculate_vanilla_latency()  # fills in latency_after_me for all submodules
            self.calculate_ramp_latency()  # fills in ramp_latencies_up_until_me for all submodules
    
    def get_all_childrens_fullname(self):
        if self.children == []:
            return [self.full_name]
        return [self.full_name] + list(itertools.chain.from_iterable([c.get_all_childrens_fullname() for c in self.children]))
    
    def get_child_with_name(self, names: list):
        if names == []:  # found child
            return self
        assert self.children != []
        for child in self.children:
            if child.name == names[0]:
                return child.get_child_with_name(names[1:])
        return None

    def __str__(self) -> str:
        # indent = "├─" * self.depth
        indent = "\t" * self.depth
        assert self.fwd_latency_pctg is not None, f"self.fwd_latency_pctg is None for module {self.name}"
        # NOTE(ruipan): self.fwd_latency_pctg should be filled in -- if not, comment out third line
        curr_str = (f"{indent}({self.name}): {self.type}, num_params {self.num_params}, " \
                    f"{round(self.fwd_latency, 3)}ms, " \
                    f"{round(self.fwd_latency_pctg, 3)}% latency, " \
                    f"full_name {self.full_name}, " \
                    f"fwd_latency_orig {self.fwd_latency_orig}ms, " \
                    f"vanilla_latency_up_until_me {self.vanilla_latency_up_until_me}ms, " \
                    f"ramp_latencies_up_until_me {self.ramp_latencies_up_until_me}, " \
                    f"input shape {self.input_shape}, output shape {self.output_shape}\n")
        for child in self.children:
            curr_str += str(child)
        return curr_str

    def __add__(self, other):
        """Aggregates the [fwd_latency, macs, fwd_flops] of two profile objects.
        Assumes that the sturcture is identical and only the fields above are 
        different.

        Args:
            other (Profile): profile object

        Returns:
            Profile: profile object
        """
        assert self.type == other.type, "only supports addition of profiles of the same type!"
        assert self.depth == other.depth, "depth of profile objects within the model should be the same!"
        assert self.num_params == other.num_params, "num_params of profile objects within the model should be the same!"
        assert self.input_shape == other.input_shape, "input_shape of profile objects within the model should be the same!"
        assert self.output_shape == other.output_shape, "output_shape of profile objects within the model should be the same!"
        assert self.input_elem_bytes == other.input_elem_bytes, "input_elem_bytes of profile objects within the model should be the same!"
        assert self.output_elem_bytes == other.output_elem_bytes, "output_elem_bytes of profile objects within the model should be the same!"
        
        if self.name != other.name:  # top-level profiles to be added, use temp name
            # this only happens when taking the average of transformer layers
            new_name = "aggregated_profile"
        else:  # children of top-level profiles to be added, keep name
            new_name = self.name        
        return Profile(
            name=new_name,
            full_name=self.full_name,
            type=self.type,
            depth=self.depth,
            num_params=self.num_params,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            input_elem_bytes=self.input_elem_bytes,
            output_elem_bytes=self.output_elem_bytes,
            fwd_latency=self.fwd_latency + other.fwd_latency,
            fwd_latency_pctg=self.fwd_latency_pctg + other.fwd_latency_pctg,
            macs=self.macs + other.macs,
            fwd_flops=self.fwd_flops + other.fwd_flops,
            children=[x + y for x, y in zip(self.children, other.children)],
            vanilla_latency_up_until_me=self.vanilla_latency_up_until_me + other.vanilla_latency_up_until_me,
            ramp_latencies_up_until_me=[x + y for x, y in zip(self.ramp_latencies_up_until_me, other.ramp_latencies_up_until_me)],
        )
    
    def __radd__(self, other):
        """accommodate __add__ to be compatible with
        python sum(), which tries to do 0 (int) + other (Profile)

        Args:
            other (Profile): other Profile object to be added

        Returns:
            Profile: latency-aggregated Profile object
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def average_transformer_layers(self, module_list: list) -> list:
        """For transformer-based models, adjust the latencies of all encoder/decoder
        layers to be the same (to not interfere with ramp selection, etc.)

        Args:
            module_list (list of Profile): list of profiles of transformer layers
        
        Returns:
            list: list of profiles with averaged latencies
        """
        num_layers = len(module_list)
        aggregated_profile = sum(module_list)  # aggregate latencies of all transformer layers
        aggregated_profile.divide_latency_by(num_layers)  # take average of latencies
        return [aggregated_profile] * num_layers
    
    def divide_latency_by(self, factor: float) -> None:
        """Divide a profile's (and all its children's) statistics
        by a factor. Helper method for average_transformer_layers().

        Args:
            factor (float): adjustment factor
        """
        self.fwd_latency /= factor
        self.fwd_latency_orig /= factor
        self.fwd_latency_pctg /= factor
        self.macs /= factor
        self.fwd_flops /= factor
        self.vanilla_latency_up_until_me /= factor
        self.ramp_latencies_up_until_me = [x / factor for x in self.ramp_latencies_up_until_me]
        for child in self.children:
            child.divide_latency_by(factor)

    def calibrate_latency(self) -> float:
        """Recursively calibrates the forward latencies of all modules
        in the model. This prevents errors that affects latency calculations,
        such as:
        * Lists of modules (e.g. Sequential) with 0ms latency
        * Lists of modules (e.g. Sequential) with latency unequal to the 
        sum of all its child modules

        Returns:
            float: calibrated forward latency of a module
        """
        if self.children == []:  # compute modules, latency cannot go wrong
            pass
        else:
            # make parent's latency equal to the sum of all children's latency
            # so that we don't have miscalculations in savings, etc.
            sum_children_latencies = sum([
                child.calibrate_latency() for child in self.children
            ])
            self.fwd_latency = sum_children_latencies
        # for bert models only
        # NOTE(ruipan): although unlikely, this "if" logic might be a pitfall if cv models have the exact same structure
        if self.name == "layer" and self.type == "ModuleList":
            # make all encoder layers have the same latency. we do this so that
            # ramp selection is not interfered by minor latency differences between
            # encoder/decoder layers.
            updated_module_list = self.average_transformer_layers(self.children)
            for layer_id, transformer_layer in enumerate(updated_module_list):
                transformer_layer = copy.deepcopy(transformer_layer)
                transformer_layer.name = str(layer_id)
                self.children[layer_id] = transformer_layer
        return self.fwd_latency

    def normalize_latency(self, normalization_factor: float) -> float:
        """Recursively normalizes the forward latencies of all modules
        in the model, such that the overall serving latency remains
        unchanged after calibrate_latency.
        
        Args:
            normalization_factor (float): original forward latency of model 
            divided by calibrated forward latency of model. All modules with
            nonzero forward latencies should do this normalization.
        """
        self.fwd_latency *= normalization_factor
        for c in self.children:
            c.normalize_latency(normalization_factor)
        
    def calculate_overall_stats(self, total_duration: float, parents_name: str) -> None:
        """Recursively fills in the overall percentage statistics
        and the layer full names

        Args:
            total_duration (float): total duration of one fwd
            pass of the model in ms
            parents_name (str): all parent modules' names,
            joined by '.'
        """
        # if self.type in ["ModuleList", "Sequential"] and self.fwd_latency == 0:
        if self.fwd_latency == 0:
            # latency is 0, aggregate latencies from children first
            self.fwd_latency = sum([c.fwd_latency for c in self.children])

        if parents_name == "model":
            self.full_name = self.name
        else:
            self.full_name = f"{parents_name}.{self.name}"

        self.fwd_latency_pctg = 100 * self.fwd_latency / total_duration
        for child in self.children:
            child.calculate_overall_stats(
                total_duration=total_duration, parents_name=self.full_name)

    def calculate_vanilla_latency(self, latency_before_me: float = 0.0,
                                  in_ramp: bool = False) -> float:
        """Recursively calculates the latency after data passes
        through each child submodule, disregarding the overhead of EE ramps. 
        Requires the ordering of a model profile to be sequential 
        (i.e. reflective of the ordering of compute).

        Args:
            latency_before_me (float, optional): latency accrued before
            data passes through the current submodule. Defaults to 0.0.
            in_ramp (bool, optinal): whether current submodule is part 
            of a ramp.

        Returns:
            float: latency after data passes the current submodule.
        """
        if self.children == []:  # compute modules like Conv2d or BatchNorm2d
            if self.name == "branch_net" or in_ramp:
                self.vanilla_latency_up_until_me = latency_before_me
            else:
                self.vanilla_latency_up_until_me = latency_before_me + self.fwd_latency
            return self.vanilla_latency_up_until_me
        else:  # lists of modules like Sequential or DistillerBottleneck
            for child_module in self.children:
                latency_after_child = child_module.calculate_vanilla_latency(
                    latency_before_me,
                    in_ramp=(self.name == "branch_net") or in_ramp,
                )
                latency_before_me = latency_after_child
            self.vanilla_latency_up_until_me = self.children[-1].vanilla_latency_up_until_me
            return self.vanilla_latency_up_until_me

    def calculate_ramp_latency(self, ramp_latencies_up_until_me: list = []) -> list:
        """For each ramp, recursively fills in the latencies of all ramps
        prior to that ramp

        Args:
            ramp_latencies_up_until_me (list, optional): list of latencies of
            prior ramps. Defaults to [].

        Returns:
            list: list of latencies of prior ramps, plus the current ramp's
            latency if the current module is a ramp
        """
        if self.name == "branch_net":
            self.ramp_latencies_up_until_me = ramp_latencies_up_until_me + \
                [self.fwd_latency]
            return self.ramp_latencies_up_until_me
        for child in self.children:
            ramp_latencies_up_until_me = child.calculate_ramp_latency(
                ramp_latencies_up_until_me)
        return ramp_latencies_up_until_me

    def get_all_children_with_type(self, type: str) -> list:
        """Recursively finds all child modules of current module
        with a specific type, and returns them in a list

        Args:
            type (str): type of torch.nn module

        Returns:
            list: list of Profile objects
        """
        all_children_with_type = []
        for child in self.children:
            if child.type == type:
                all_children_with_type.append(child)
            all_children_with_type += child.get_all_children_with_type(type)
        return all_children_with_type
    
    def get_all_children_with_name(self, name: str) -> list:
        """Recursively finds all child modules of current module
        with a specific name, and returns them in a list

        Args:
            name (str): name of torch.nn module

        Returns:
            list: list of Profile objects
        """
        all_children_with_name = []
        for child in self.children:
            if child.name == name:
                all_children_with_name.append(child)
            all_children_with_name += child.get_all_children_with_name(name)
        return all_children_with_name

    def get_layer_output_size(self, layer_output_size: dict = {}) -> dict:
        # TODO(ruipan)
        layer_output_size.update({
            self.full_name: self.output_shape
        })
        for child in self.children:
            child.get_layer_output_size(layer_output_size)
        return layer_output_size

    # def get_latency_savings(self) -> tuple[float, list]:
    #     """For an EE model, calculate the percentage of latency savings
    #     when a sample exits through each ramp.

    #     Raises:
    #         NotImplementedError: currently, only support DeeBERT and
    #         EE models where all modules are sequentially traversed

    #     Returns:
    #         tuple[float, list]: forward latency of the whole model and a
    #         list, in which index x represents the percentage of latency
    #         savings a sample gets from exiting through ramp x.
    #     """
    #     assert self.name == "model", \
    #         f"Getting latency savings on child of a model!"
    #     overall_latency = self.fwd_latency

    #     if self.type == "ResNetEarlyExit":
    #         # ... or all models where modules are sequentially traversed
    #         self.calculate_vanilla_latency()  # fills in latency_after_me for all submodules
    #         self.calculate_ramp_latency()  # fills in ramp_latencies_up_until_me for all submodules
    #         all_branchpoints = self.get_all_children_with_type("BranchPoint")
    #         latency_savings = [
    #             (1 - bp.vanilla_latency_up_until_me / overall_latency) * 100
    #             for bp in all_branchpoints
    #         ]
    #     elif "BertForSequenceClassification" in self.type:
    #         # this assumes bert-EE is structured like this:
    #         # embeddings -> layer0 -> highway0 -> layer1 -> highway1 -> ...
    #         embeddings_latency = self.children[0].children[0].fwd_latency
    #         if "Distil" in self.type:
    #             all_bert_layers = self.get_all_children_with_type("TransformerBlock")
    #         else:
    #             all_bert_layers = self.get_all_children_with_type("BertLayer")
    #         all_bert_highways = self.get_all_children_with_type("BertHighway")
    #         avg_layer_latency = sum([x.fwd_latency for x in all_bert_layers]) / len(all_bert_layers)
    #         avg_highway_latency = sum([x.fwd_latency for x in all_bert_highways]) / len(all_bert_highways)

    #         latency_savings = []
    #         for i in range(1, len(all_bert_highways) + 1):
    #             latency_savings.append(100 * (
    #                 1 - (embeddings_latency + i * (avg_layer_latency + avg_highway_latency)) / overall_latency
    #             ))
    #     else:
    #         raise NotImplementedError(f"Latency saving calculation for model type {self.type} not implemented!")

    #     return overall_latency, latency_savings


def calculate_size(
    # inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: int | None  # "|" is python 3.10+ only
    inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: int = None
):
    """
    Set input_size or output_size using the model's inputs.
    Returns the corrected shape of `inputs` and the size of
    a single element in bytes.
    NOTE(ruipan): modified from https://github.com/TylerYep/torchinfo/blob/main/torchinfo/layer_info.py#L88
    """
    if inputs is None:
        size, elem_bytes = [], 0

    # pack_padded_seq and pad_packed_seq store feature into data attribute
    elif (
        isinstance(inputs, (list, tuple)) and inputs and hasattr(
            inputs[0], "data")
    ):
        size = list(inputs[0].data.size())
        elem_bytes = inputs[0].data.element_size()
        if batch_dim is not None:
            size = size[:batch_dim] + [1] + size[batch_dim + 1:]

    elif isinstance(inputs, dict):
        output = list(inputs.values())[-1]
        size, elem_bytes = nested_list_size(output)
        if batch_dim is not None:
            size = [size[:batch_dim] + [1] + size[batch_dim + 1:]]

    elif isinstance(inputs, torch.Tensor):
        size = list(inputs.size())
        elem_bytes = inputs.element_size()

    elif isinstance(inputs, (list, tuple)):
        size, elem_bytes = nested_list_size(inputs)
        if batch_dim is not None and batch_dim < len(size):
            size[batch_dim] = 1

    else:
        raise TypeError(
            "Model contains a layer with an unsupported input or output type: "
            f"{inputs}, type: {type(inputs)}"
        )

    return size, elem_bytes


# def nested_list_size(inputs: Sequence[Any] | torch.Tensor) -> tuple[list[int], int]:  # "|" is python 3.10+ only
def nested_list_size(inputs):
    """
    Flattens nested list size.
    NOTE(ruipan): copied from https://github.com/TylerYep/torchinfo/blob/main/torchinfo/layer_info.py#L312
    """
    if hasattr(inputs, "tensors"):
        size, elem_bytes = nested_list_size(inputs.tensors)
    elif isinstance(inputs, torch.Tensor):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif not hasattr(inputs, "__getitem__") or not inputs:
        size, elem_bytes = [], 0
    elif isinstance(inputs, dict):
        size, elem_bytes = nested_list_size(list(inputs.values()))
    elif (
        hasattr(inputs, "size")
        and callable(inputs.size)
        and hasattr(inputs, "element_size")
        and callable(inputs.element_size)
    ):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif isinstance(inputs, (list, tuple)):
        size, elem_bytes = nested_list_size(inputs[0])
    else:
        size, elem_bytes = [], 0

    return size, elem_bytes


class TIDSProfiler(object):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.
    The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.
    When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file and no user code change is required.

    If using the profiler as a standalone package, one imports the flops_profiler package and use the APIs.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = TIDSProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops(as_string=True)
                    params = prof.get_total_params(as_string=True)
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self, model, ds_engine=None):
        self.model = model
        self.ds_engine = ds_engine
        self.started = False
        self.func_patched = False

    def generate_profile(self, module=None, name="model", curr_depth=0):
        """Generates profiling information of a model

        Args:
            module (torch.mm.Module, optional): Module to be profiled. 
            Defaults to None.
            name (str, optional): Name of the module. Defaults to "model".
            curr_depth (int, optional): Current depth in the model. Note
            that this depth is not horizontal depth. Defaults to 0.

        Returns:
            Profile: profiling result
        """
        if module is None:
            module = self.model

        # if module.__class__.__name__ in ["ModuleList", "ReLU", "Sequential"]:
        if not hasattr(module, "__input_shape__"):
            # post_hook is not triggered for ModuleList, so these
            # module attributes are never set
            input_shape, output_shape = None, None
            input_elem_bytes, output_elem_bytes = None, None
        else:
            input_shape, output_shape = module.__input_shape__, module.__output_shape__
            input_elem_bytes, output_elem_bytes = module.__input_elem_bytes__, module.__output_elem_bytes__

        profile = Profile(
            name=name,
            full_name=name,
            type=module.__class__.__name__,
            depth=curr_depth,
            num_params=module.__params__,
            input_shape=input_shape,
            output_shape=output_shape,
            input_elem_bytes=input_elem_bytes,
            output_elem_bytes=output_elem_bytes,
            fwd_latency=module.__duration__*1000,  # s -> ms
            fwd_latency_pctg=100.0,
            macs=module.__macs__,
            fwd_flops=module.__flops__,
        )

        child_profiles = []
        for child_name, child_module in module.named_children():
            # NOTE(ruipan): module.(named_){modules,children} returns {all modules,immediate child modules}
            child_profile = self.generate_profile(
                module=child_module, name=child_name, curr_depth=curr_depth+1
            )
            child_profiles.append(child_profile)
        profile.set_child_modules(child_profiles)
        return profile

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_functionals()
        _patch_tensor_methods()

        def register_module_hooks(module, ignore_list):
            # print(f"register_module_hooks invoked for module of type {type(module)}")
            # print(f"hasattr(module, '__post_hook_handle__'): {hasattr(module, '__post_hook_handle__')}")
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(
                        MODULE_HOOK_MAPPING[type(module)])
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(
                    pre_hook)

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([elem[1]
                                            for elem in module_flop_count[-1]])
                    module_flop_count.pop()
                    module.__macs__ += sum([elem[1]
                                           for elem in module_mac_count[-1]])
                    module_mac_count.pop()

                if not hasattr(module, "__input_shape__"):
                    size, elem_bytes = calculate_size(input)
                    module.__input_shape__ = size
                    module.__input_elem_bytes__ = elem_bytes
                    # print(f"post_hook invoked for module of type {type(module)}, __input_shape__: {module.__input_shape__}")

                if not hasattr(module, "__output_shape__"):
                    size, elem_bytes = calculate_size(output)
                    module.__output_shape__ = size
                    module.__output_elem_bytes__ = elem_bytes

            if not hasattr(module, "__post_hook_handle__"):
                # print(f"post_hook registered for module of type {type(module)}")
                module.__post_hook_handle__ = module.register_forward_hook(
                    post_hook)

            def start_time_hook(module, input):
                if deepspeed_imported:
                    get_accelerator().synchronize()
                elif cuda_available:
                    torch.cuda.synchronize("cuda:0")
                module.__start_time__ = time.time()

            if not hasattr(module, "__start_time_hook_handle"):
                module.__start_time_hook_handle__ = module.register_forward_pre_hook(
                    start_time_hook)

            def end_time_hook(module, input, output):
                if deepspeed_imported:
                    get_accelerator().synchronize()
                elif cuda_available:
                    torch.cuda.synchronize("cuda:0")
                module.__duration__ += time.time() - module.__start_time__
                # print(f"{type(module)}: module.__duration__  updated to {module.__duration__ }")

            if not hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__ = module.register_forward_hook(
                    end_time_hook)

        self.model.apply(
            partial(register_module_hooks, ignore_list=ignore_list))
        self.started = True
        self.func_patched = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, "__start_time_hook_handle__"):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__

        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())
            module.__start_time__ = 0
            module.__duration__ = 0

        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__macs__"):
                del module.__macs__
            if hasattr(module, "__params__"):
                del module.__params__
            if hasattr(module, "__start_time__"):
                del module.__start_time__
            if hasattr(module, "__duration__"):
                del module.__duration__
            if hasattr(module, "__input_shape__"):
                del module.__input_shape__
            if hasattr(module, "__output_shape__"):
                del module.__output_shape__
            if hasattr(module, "__input_elem_bytes__"):
                del module.__input_elem_bytes__
            if hasattr(module, "__output_elem_bytes__"):
                del module.__output_elem_bytes__

        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model)
        return num_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = get_module_macs(self.model)
        return macs_to_string(total_macs) if as_string else total_macs

    def get_total_duration(self, as_string=False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The latency of the model forward pass.
        """
        total_duration = get_module_duration(self.model)
        return duration_to_string(total_duration) if as_string else total_duration

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return params_to_string(
            self.model.__params__) if as_string else self.model.__params__

    def print_model_profile(self,
                            profile_step=1,
                            module_depth=-1,
                            top_modules=1,
                            detailed=True,
                            output_file=None):
        """Prints the model graph with the measured profile attached to each module.

        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
        """
        if not self.started:
            return
        import sys
        import os.path
        original_stdout = None
        f = None
        if output_file and output_file != "":
            dir_path = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, "w")
            sys.stdout = f

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        print(
            "\n-------------------------- DeepSpeed Flops TIDSProfiler --------------------------"
        )
        print(f'Profile Summary at step {profile_step}:')
        print(
            "Notations:\ndata parallel size (dp_size), model parallel size(mp_size),\nnumber of parameters (params), number of multiply-accumulate operations(MACs),\nnumber of floating-point operations (flops), floating-point operations per second (FLOPS),\nfwd latency (forward propagation latency), bwd latency (backward propagation latency),\nstep (weights update latency), iter latency (sum of fwd, bwd and step latency)\n"
        )
        if self.ds_engine:
            print('{:<60}  {:<8}'.format(
                'world size: ', self.ds_engine.world_size))
            print('{:<60}  {:<8}'.format('data parallel size: ',
                                         self.ds_engine.dp_world_size))
            print('{:<60}  {:<8}'.format('model parallel size: ',
                                         self.ds_engine.mp_world_size))
            print('{:<60}  {:<8}'.format(
                'batch size per GPU: ',
                self.ds_engine.train_micro_batch_size_per_gpu()))

        print('{:<60}  {:<8}'.format(
            'params per gpu: ', params_to_string(total_params)))
        print('{:<60}  {:<8}'.format(
            'params of model = params per GPU * mp_size: ',
            params_to_string(total_params *
                             ((self.ds_engine.mp_world_size) if self.ds_engine else 1))))

        print('{:<60}  {:<8}'.format(
            'fwd MACs per GPU: ', macs_to_string(total_macs)))

        print('{:<60}  {:<8}'.format(
            'fwd flops per GPU: ', num_to_string(total_flops)))

        print('{:<60}  {:<8}'.format(
            'fwd flops of model = fwd flops per GPU * mp_size: ',
            num_to_string(total_flops *
                          ((self.ds_engine.mp_world_size) if self.ds_engine else 1))))

        fwd_latency = self.get_total_duration()
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            fwd_latency = self.ds_engine.timers(
                'forward').elapsed(False) / 1000.0
        print('{:<60}  {:<8}'.format(
            'fwd latency: ', duration_to_string(fwd_latency)))
        print('{:<60}  {:<8}'.format(
            'fwd FLOPS per GPU = fwd flops per GPU / fwd latency: ',
            flops_to_string(total_flops / fwd_latency)))

        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            bwd_latency = self.ds_engine.timers(
                'backward').elapsed(False) / 1000.0
            step_latency = self.ds_engine.timers(
                'step').elapsed(False) / 1000.0
            print('{:<60}  {:<8}'.format('bwd latency: ',
                                         duration_to_string(bwd_latency)))
            print('{:<60}  {:<8}'.format(
                'bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency: ',
                flops_to_string(2 * total_flops / bwd_latency)))
            print('{:<60}  {:<8}'.format(
                'fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency): ',
                flops_to_string(3 * total_flops / (fwd_latency + bwd_latency))))

            print('{:<60}  {:<8}'.format('step latency: ',
                                         duration_to_string(step_latency)))

            iter_latency = fwd_latency + bwd_latency + step_latency
            print('{:<60}  {:<8}'.format('iter latency: ',
                                         duration_to_string(iter_latency)))
            print('{:<60}  {:<8}'.format(
                'FLOPS per GPU = 3 * fwd flops per GPU / iter latency: ',
                flops_to_string(3 * total_flops / iter_latency)))

            samples_per_iter = self.ds_engine.train_micro_batch_size_per_gpu(
            ) * self.ds_engine.world_size
            print('{:<60}  {:<8.2f}'.format('samples/second: ',
                                            samples_per_iter / iter_latency))

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [
                params_to_string(params),
                "{:.2%} Params".format(
                    params / total_params if total_params else 0),
                macs_to_string(macs),
                "{:.2%} MACs".format(0.0 if total_macs ==
                                     0 else macs / total_macs),
            ]
            duration = get_module_duration(module)

            items.append(duration_to_string(duration))
            items.append(
                "{:.2%} latency".format(0.0 if total_duration == 0 else duration /
                                        total_duration))
            items.append(flops_to_string(
                0.0 if duration == 0 else flops / duration))
            items.append(module.original_extra_repr())
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        print(
            "\n----------------------------- Aggregated Profile per GPU -----------------------------"
        )
        self.print_model_aggregated_profile(module_depth=module_depth,
                                            top_modules=top_modules)

        if detailed:
            print(
                "\n------------------------------ Detailed Profile per GPU ------------------------------"
            )
            print(
                "Each module profile is listed after its name in the following order: \nparams, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS"
            )
            print(
                "\nNote: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.\n2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.\n3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.\n"
            )
            print(self.model)

        self.model.apply(del_extra_repr)

        print(
            "------------------------------------------------------------------------------"
        )

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 1.
        """
        info = {}
        if not hasattr(self.model, "__flops__"):
            print(
                "no __flops__ attribute in the model, call this function after start_profile and before end_profile"
            )
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                ]  # macs, params, time
            info[curr_depth][module.__class__.__name__][0] += get_module_macs(
                module)
            info[curr_depth][module.__class__.__name__][1] += module.__params__
            info[curr_depth][module.__class__.__name__][2] += get_module_duration(
                module)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)

        walk_module(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1

        print(
            f'Top {top_modules} modules in terms of params, MACs or fwd latency at different model depths:'
        )

        for d in range(depth):
            num_items = min(top_modules, len(info[d]))

            sort_macs = {
                k: macs_to_string(v[0])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][0],
                            reverse=True)[:num_items]
            }
            sort_params = {
                k: params_to_string(v[1])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][1],
                            reverse=True)[:num_items]
            }
            sort_time = {
                k: duration_to_string(v[2])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][2],
                            reverse=True)[:num_items]
            }

            print(f"depth {d}:")
            print(f"    params      - {sort_params}")
            print(f"    MACs        - {sort_macs}")
            print(f"    fwd latency - {sort_time}")


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs


def _relu_flops_compute(input, inplace=False):
    return input.numel(), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel(), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return input.numel(), 0


def _leaky_relu_flops_compute(input: Tensor,
                              negative_slope: float = 0.01,
                              inplace: bool = False):
    return input.numel(), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _gelu_flops_compute(input, **kwargs):
    return input.numel(), 0


def _pool_flops_compute(input,
                        kernel_size,
                        stride=None,
                        padding=0,
                        dilation=None,
                        ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=None,
                        return_indices=None):
    return input.numel(), 0


def _conv_flops_compute(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(
        _prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(
        _prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: Optional[Tensor] = None,
                              bias: Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _upsample_flops_compute(input, **kwargs):
    size = kwargs.get('size', None)
    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    scale_factor = kwargs.get('scale_factor', None)
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel(), 0


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def wrapFunc(func, funcFlopCompute):
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return oldFunc(*args, **kwds)

    newFunc.__str__ = func.__str__

    return newFunc


def _patch_functionals():
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(
        F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(
        F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(
        F.conv_transpose3d, _conv_trans_flops_compute)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(
        F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(
        F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(
        F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(
        F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(
        F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(
        F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)


def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.Tensor.bmm, _matmul_flops_compute)

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(
        torch.Tensor.addmm, _tensor_addmm_flops_compute)

    torch.mul = wrapFunc(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute)

    torch.add = wrapFunc(torch.add, _add_flops_compute)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute)

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute)

    torch.baddbmm = wrapFunc(torch.baddbmm, _tensor_addmm_flops_compute)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.embedding = old_functions[F.embedding.__str__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    torch.mm = old_functions[torch.mm.__str__]
    torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    torch.bmm = old_functions[torch.matmul.__str__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]

    torch.einsum = old_functions[torch.einsum.__str__]

    torch.baddbmm = old_functions[torch.baddbmm.__str__]


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}


def num_to_string(num, precision=2):
    if num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + " G"
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + " M"
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + " K"
    else:
        return str(num)


def macs_to_string(macs, units=None, precision=2):
    if units is None:
        if macs // 10**9 > 0:
            return str(round(macs / 10.0**9, precision)) + " GMACs"
        elif macs // 10**6 > 0:
            return str(round(macs / 10.0**6, precision)) + " MMACs"
        elif macs // 10**3 > 0:
            return str(round(macs / 10.0**3, precision)) + " KMACs"
        else:
            return str(macs) + " MACs"
    else:
        if units == "GMACs":
            return str(round(macs / 10.0**9, precision)) + " " + units
        elif units == "MMACs":
            return str(round(macs / 10.0**6, precision)) + " " + units
        elif units == "KMACs":
            return str(round(macs / 10.0**3, precision)) + " " + units
        else:
            return str(macs) + " MACs"


def number_to_string(num, units=None, precision=2):
    if units is None:
        if num // 10**9 > 0:
            return str(round(num / 10.0**9, precision)) + " G"
        elif num // 10**6 > 0:
            return str(round(num / 10.0**6, precision)) + " M"
        elif num // 10**3 > 0:
            return str(round(num / 10.0**3, precision)) + " K"
        else:
            return str(num) + " "
    else:
        if units == "G":
            return str(round(num / 10.0**9, precision)) + " " + units
        elif units == "M":
            return str(round(num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(num / 10.0**3, precision)) + " " + units
        else:
            return str(num) + " "


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**12 > 0:
            return str(round(flops / 10.0**12, precision)) + " TFLOPS"
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GFLOPS"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MFLOPS"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KFLOPS"
        else:
            return str(flops) + " FLOPS"
    else:
        if units == "TFLOPS":
            return str(round(flops / 10.0**12, precision)) + " " + units
        if units == "GFLOPS":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MFLOPS":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KFLOPS":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " FLOPS"


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)


def duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + " s"
        elif duration * 10**3 > 1:
            return str(round(duration * 10**3, precision)) + " ms"
        elif duration * 10**6 > 1:
            return str(round(duration * 10**6, precision)) + " us"
        else:
            return str(duration)
    else:
        if units == "us":
            return str(round(duration * 10.0**6, precision)) + " " + units
        elif units == "ms":
            return str(round(duration * 10.0**3, precision)) + " " + units
        else:
            return str(round(duration, precision)) + " s"

    # can not iterate over all submodules using self.model.modules()
    # since modules() returns duplicate modules only once


def get_module_flops(module):
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_flops(child)
    return sum


def get_module_macs(module):
    sum = module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_macs(child)
    return sum


def get_module_duration(module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for m in module.children():
            duration += m.__duration__
    return duration


def get_model_profile(
    model,
    input_shape=None,
    args=[],
    kwargs={},
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=1,
    as_string=True,
    output_file=None,
    ignore_modules=None,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = TIDSProfiler(model)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape,
                 ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input]
    assert (len(args) > 0) or (len(kwargs) >
                               0), "args and/or kwargs must be specified if input_shape is None"

    for _ in range(warm_up):
        if kwargs:
            _ = model(*args, **kwargs)
        else:
            _ = model(*args)
    prof.start_profile(ignore_list=ignore_modules)

    if kwargs:
        _ = model(*args, **kwargs)
    else:
        _ = model(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params
