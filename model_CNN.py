import torch
import torch.nn as nn
from dataset import *
import torch.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_channels_new,
        num_classes
    ):
        super().__init__()
        self.batnorm = nn.BatchNorm1d(num_features=2)
        self.batnorm1 = nn.BatchNorm1d(num_features=64)
        self.batnorm2 = nn.BatchNorm1d(num_features=32)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels_new,
            kernel_size=3,
            padding="same",
        )
        self.den1 = nn.Linear(in_features=128, out_features=64)
        self.den2 = nn.Linear(in_features=64, out_features=32)
        self.classification = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, labels=None):
        b = x.size(0)
        x = self.batnorm(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batnorm1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.den1(x)
        x = self.batnorm2(x)
        x = self.den2(x)
        x = self.batnorm2(x)
        x = x.view(b, -1)
        x = self.classification(x)
        logits = self.softmax(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return logits

# mod = CNN(in_channels=2,
#         out_channels=64,
#         out_channels_new=32,
#         num_classes=220)  
# x = torch.randn(32, 2, 128)
# y = mod(x)
# print(y)