from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
import PIL.Image as Image

import pandas as pd


class Dataset(nn.Dataset):
    def __init__(self):
        self.dataset = Config.dataset
        self.data = []
        self.label = []
        self.data_root = None
        self.check_exists()
        self.load_data()

    def check_exists(self):
        if self.dataset == 'CICIDS2017':
            data_path = '../dataset/CICIDS2017/MachineLearningCSV/preprocessed'

        elif self.dataset == 'CICDDoS2019':
            data_path = '../dataset/CICDDoS2019/preprocessed'

        if not os.path.exists(data_path):
            # TODO: download dataset

        self.data_root = data_path

    def load_data(self):

        data = pd.read_csv(os.path.join(self.data_root, 'data.csv.gz'), compression='gzip')

        for file in data:
            # self.data.append(file)
            # self.label.append(file.split('_')[0])
            self.data.append(file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
