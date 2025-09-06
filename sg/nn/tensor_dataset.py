from shkyegrad import Tensor

class TensorDataset:
    def __init__(self, x: Tensor, y: Tensor):
        assert len(x) == len(y)

        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

