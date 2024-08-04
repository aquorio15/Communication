import base64
import json
import os
import cv2
import pandas as pd
import pickle
import sys
from PIL import Image
import numpy as np
import yaml
import torch
import sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)
#%%
class Communicationdataset:
    
    def __init__(self, root: str):
        self.root = root
        self.data = pickle.load(
            open(f"{self.root}/communication_data_20.pkl", "rb")
        )
        self.label = pickle.load(
            open(f"{self.root}/communication_label_20.pkl", "rb")
        )
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        label = self.label[item]
        return datum, label

# Train_Dataset = Communicationdataset(
#     root="/DATA/nfsshare/Amartya/EMNLP-WACV/communication_journal",
# )
# train_size = int(0.7 * len(Train_Dataset))
# valid_size = len(Train_Dataset) - train_size
# trainset, testset = torch.utils.data.random_split(
#             Train_Dataset, [train_size, valid_size]
#         )
# train_sampler = (
#         RandomSampler(trainset)
#     )
# test_sampler = SequentialSampler(testset)
# train_loader = DataLoader(
#         trainset,
#         sampler=train_sampler,
#         batch_size=32,
#         num_workers=8,
#         drop_last=False,
#         pin_memory=True,
#     )
# a, b = next(iter(train_loader))
# print(a.shape)
# print(b)