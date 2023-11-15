from config import Config

import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CIDDataset(Dataset):
    def __init__(self, dataset_name='chd'):
        super(Dataset, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_path = Config.data_root

        self.images = []
        self.labels = []

        self.get_data_from_dir()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]
        lb = self.labels[idx]

        img = Image.open(img).convert('L')
        img = transforms.ToTensor()(img)

        lb = torch.tensor(lb)

        return {'input': img, 'label': lb}

    def get_data_from_dir(self):

        attacks = os.listdir(self.dataset_path)  # dos, fuzzy, ...

        for attack in attacks:

            path_to_data = os.path.join(self.dataset_path, attack)  # ../dataset/CHD/id_image_64/fuzzy
            filenames = os.listdir(path_to_data)  # abnormal_22.png, normal_26237.png, ...

            for file in filenames:  # normal_0.png

                full_path = os.path.join(path_to_data, file)  # # ../dataset/CHD/id_image_64/fuzzy/normal_0.png

                label = file.split('_')[0]

                self.labels.append(int(label))
                self.images.append(full_path)