from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import PIL.Image as Image


class Dataset(nn.Dataset):
    def __init__(self):
        self.data_root = Config.data_root
        self.data = []
        self.label = []
        self.load_data()

    def load_data(self):
        data = os.listdir(self.data_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]