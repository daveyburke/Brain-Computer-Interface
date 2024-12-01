import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class EEGModel(nn.Module):
    """
    Based on https://arxiv.org/pdf/1611.08024
    See also review paper: https://www.sciencedirect.com/science/article/abs/pii/S1746809420303116
    Applies temporal filters, then spatial filters, then conv, then dense
    Example input (64 <batch>, 1 <conv_channel>, 3 <eeg_channels:C>, 256 <time_series:T>)
    """
    def __init__(self):
        super(EEGModel, self).__init__()

        # Layer 1 (8x temporal filters within each EEG channel)
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding = (0, 32), bias=False)  # [1, C, T] -> [8, C, T]
        self.batch_norm1 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.9)

        # Layer 2 (spatial, 2x depthwise/vertical convolutions across 3 EEG channels)
        self.conv2 = weight_norm(nn.Conv2d(8, 16, (3, 1), groups = 8, bias=False))  # [8, C, T] -> [16, 1, T]
        self.batch_norm2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9)  # eps/momentum to keep stable
        self.avg_pool1 = nn.AvgPool2d((1, 4))  # [16, 1, T // 4]
        self.dropout = nn.Dropout2d(0.25)

        # Layer 3 (pointwise conv, for efficiency like MobileNet)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding = (0, 8), groups = 16, bias=False) # [16, 1, T // 4]
        self.conv4 = nn.Conv2d(16, 16, (1, 1))
        self.batch_norm3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9)
        self.avg_pool2 = nn.AvgPool2d((1, 8))  # [16, 1, T // 32]

        # Layer 4 (dense layer)
        self.flatten = nn.Flatten()  # 16 * 256 // 32
        self.dense = nn.Linear(16 * 256 // 32, 2)
        
    def forward(self, x):
        # Layer 1
        out = self.conv1(x)
        out = self.batch_norm1(out)

        # Layer 2
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.elu(out)
        out = self.avg_pool1(out)
        out = self.dropout(out)

        # Layer 3
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.batch_norm3(out)
        out = F.elu(out)
        out = self.avg_pool2(out)
        out = self.dropout(out)

        # Layer 4
        out = self.flatten(out)
        out = self.dense(out)

        return out, F.softmax(out, dim=1)  # logits, probabilities
    
    def apply_max_norm(self):
        # Values from https://github.com/vlawhern/arl-eegmodels
        torch.nn.utils.clip_grad_norm_(self.conv2.weight, 1.)
        torch.nn.utils.clip_grad_norm_(self.dense.weight, 0.25)
        
