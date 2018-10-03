import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .resnext_features import resnext101_32x4d_features
from .resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, feature_dim):
        super(ResNeXt101_32x4d, self).__init__()
        self.feature_dim = feature_dim
        self.features = resnext101_32x4d_features
        # face
        self.layer1x1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.face_feature = nn.Linear(12544, feature_dim)
        self.drop2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.features(input)
        # face
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.face_feature(x)
        x = self.drop2(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, feature_dim):
        super(ResNeXt101_64x4d, self).__init__()
        self.feature_dim = feature_dim
        self.features = resnext101_64x4d_features
        # face
        self.layer1x1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.face_feature = nn.Linear(12544, feature_dim)
        self.drop2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.features(input)
        # face
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.face_feature(x)
        x = self.drop2(x)
        return x


def resnext101_32x4d(**kwargs):
    model = ResNeXt101_32x4d(**kwargs)
    return model

def resnext101_64x4d(**kwargs):
    model = ResNeXt101_64x4d(**kwargs)
    return model
