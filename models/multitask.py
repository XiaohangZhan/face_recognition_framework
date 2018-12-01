import torch.nn as nn
from .ext_layers import ArcFullyConnected
from . import backbones

class MultiTaskWithLoss(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, spatial_size, arc_fc=False, feat_bn=False, s=64, m=0.5, is_pw=True, is_hard=False):
        super(MultiTaskWithLoss, self).__init__()
        self.feat_bn = feat_bn
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim, spatial_size=spatial_size)
        if feat_bn:
            self.bn1d = nn.BatchNorm1d(feature_dim, affine=False, eps=2e-5, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.arc_fc = arc_fc
            if not arc_fc:
                self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
            else:
                self.fcs = nn.ModuleList([ArcFullyConnected(feature_dim, num_classes[k], s, m, is_pw, is_hard) for k in range(self.num_tasks)])

    def forward(self, input, target=None, slice_idx=None, extract_mode=False):

        feature = self.basemodel(input)
        if self.feat_bn:
            feature = self.bn1d(feature)
        if extract_mode:
            return feature
        else:
            assert feature.size(0) == target.size(0)
            assert(len(slice_idx) == self.num_tasks + 1)
            assert slice_idx[-1] == feature.size(0), "{} vs {}".format(slice_idx[-1], feature.size(0))
            if not self.arc_fc:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            else:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...], target[k]) for k in range(self.num_tasks)]
            return [self.criterion(xx, tg) for xx, tg in zip(x, target)]
