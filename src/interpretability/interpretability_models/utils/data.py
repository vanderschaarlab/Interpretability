import torch
import pandas as pd
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X.iloc[i, :], dtype=torch.float32)
        target = self.y.iloc[i]
        return data, target


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X[i], dtype=torch.float32)
        target = torch.tensor(self.y[i], dtype=torch.float32)
        return data, target
