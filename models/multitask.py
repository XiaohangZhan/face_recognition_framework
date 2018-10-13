import torch.nn as nn
from .ext_layers import ArcFullyConnected
from . import backbones
import pdb

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

    def __init__(self, backbone, num_classes, feature_dim, spatial_size):
        super(BasicMultiTask, self).__init__()
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim, spatial_size=spatial_size)
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

class BasicMultiTaskWithLoss(nn.Module):

    def __init__(self, backbone, num_classes, feature_dim, spatial_size):
        super(BasicMultiTaskWithLoss, self).__init__()
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim, spatial_size=spatial_size)
        self.criterion = nn.CrossEntropyLoss()
        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])

    def forward(self, input, target=None, slice_idx=None, extract_mode=False):
        feature = self.basemodel(input)
        if extract_mode:
            return feature
        else:
            assert target is not None
            assert(len(slice_idx) == self.num_tasks + 1)
            x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            return [self.criterion(xx, tg) for xx, tg in zip(x, target)]

