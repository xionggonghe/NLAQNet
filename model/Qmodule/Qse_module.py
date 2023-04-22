from torch import nn
import torchvision
from torchvision import transforms
import torch.utils.data
import os
from Qconv import *
from QFunc import *
from quaternion_BN import *

class QSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(QSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            QSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)













