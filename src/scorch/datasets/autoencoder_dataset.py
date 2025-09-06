import torch

class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, x

