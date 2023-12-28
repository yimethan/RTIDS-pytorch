from config import Config
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer


class CHDDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        # Index(['Timestamp', 'CAN_ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3',
        #        'Data4', 'Data5', 'Data6', 'Data7', 'Flag'],
        #       dtype='object')
        df = pd.read_csv(Config.data_root, index_col=0)
        print(df.columns)

        imputer = SimpleImputer(strategy=Config.strategy)

        self.inp = imputer.fit_transform(df.iloc[:, :-1])
        self.labels = df.iloc[:, -1]

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):

        inp = self.inp[idx]
        lb = self.labels[idx]

        inp = torch.tensor(inp, dtype=torch.float32)
        lb = torch.tensor(lb)

        return {'input': inp, 'label': lb}
