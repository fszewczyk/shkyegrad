import torch
from typing import Tuple

class Flatten(torch.nn.Module):
    def __init__(self, start=1, end=-1):
        super().__init__()

        self.start = start
        self.end = end

    def forward(self, x):
        return x.flatten(self.start, self.end)

class Unflatten(torch.nn.Module):
    def __init__(self, dim: int, size: Tuple[int]):
        super().__init__()

        self.dim = dim
        self.size = size

    def forward(self, x):
        return x.unflatten(self.dim, self.size)
