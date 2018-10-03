import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .ext_layers import ArcFullyConnected
from . import backbones
import pdb

__all__ = ['ArcMultiTask', 'BasicMultiTask', 'MultiTaskWithPair']

class ArcMultiTask(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, arc_fc=False, s=64, m=0.5, is_pw=True, is_hard=False, extract_mode=False):
        super(ArcMultiTask, self).__init__()
        if not extract_mode:
            self.num_tasks = len(num_classes)
        self.extract_mode = extract_mode
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim)
        # arc
        if not extract_mode:
            self.arc_fc = arc_fc
            print(arc_fc, s, m, is_pw, is_hard)
            self.dropout = nn.Dropout(p=0.5)
            if not arc_fc:
                self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
            else:
                self.fcs = nn.ModuleList([ArcFullyConnected(feature_dim, num_classes[k], s, m, is_pw, is_hard) for k in range(self.num_tasks)])

    def forward(self, input, labels=None, slice_idx=None):

        feature = self.basemodel(input)
        if self.extract_mode:
            return feature
        else:
            assert(len(slice_idx) == self.num_tasks + 1)
            feature = self.dropout(feature)
            if not self.arc_fc:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            else:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...], labels[slice_idx[k]:slice_idx[k+1]]) for k in range(self.num_tasks)]
            return x

class BasicMultiTask(nn.Module):

    def __init__(self, backbone, num_classes, feature_dim):
        super(BasicMultiTask, self).__init__()
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim)
        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])

    def forward(self, input, slice_idx=None, extract_mode=False):
        feature = self.basemodel(input)
        if extract_mode:
            return feature
        else:
            assert(len(slice_idx) == self.num_tasks + 1)
            x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            return x

class MultiTaskWithPair(nn.Module):

    def __init__(self, backbone, num_classes, feature_dim):
        super(MultiTaskWithPair, self).__init__()
        self.num_tasks = len(num_classes)
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim)
        self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])

    def forward(self, input, slice_idx=None):
        assert(len(slice_idx) == self.num_tasks + 1)
        feature = self.basemodel(input)
        score = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
        return score, feature[slice_idx[-1]:, ...]

