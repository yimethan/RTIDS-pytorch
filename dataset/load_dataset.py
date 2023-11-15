import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CICIDSDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, :-3]  # Feature Columns
        self.at_type = data[:, -3]  # Attack Type Column
        self.labels = data[:, -2:]  # 1 Hot Encoded Label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]
        at_type = self.at_type[idx]

        return feature, label, at_type


def load_data():
    data = pd.read_csv('D:/dataset/cicids2017/preprocessed/data.csv.gz', compression='gzip')

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data.values, test_data.values
