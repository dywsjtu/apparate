import torch
import torch.nn as nn


class BranchPoint(nn.Module):
    """Add a branch to an existing model."""
    def __init__(self, branched_module, branch_net, timer = None):
        """
        :param branched_module: the module in the original network to which we add a branch.
        :param branch_net: the new branch
        """
        super().__init__()
        self.branched_module = branched_module
        self.branch_net = branch_net
        self.output = None
        
    def forward(self, *args, **kwargs):
        x1 = self.branched_module.forward(*args, **kwargs)
        if self.branch_net:  # NOTE(ruipan): if none, then branch_net is deactivated
            self.output = self.branch_net.forward(x1)  # deebert: self.output: (logits, pooled_output)
        return x1

class Flatten(nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

class Norm(nn.Module):
    """
    A module wrapper for vector/matrix norm
    """
    def __init__(self, p='fro', dim=None, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)


class Mean(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Mean, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        return torch.mean(x, *self.args, **self.kwargs)

class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise addition"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseSub(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise subtraction"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res -= t
        else:
            for t in input[1:]:
                res = res - t
        return res


class EltwiseMult(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise multiplication"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res


class EltwiseDiv(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise division"""
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)