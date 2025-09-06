import torch
import math

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, x):
        if self.training or self.p == 0:
            prob_to_keep = 1.0 - self.p
            mask = (torch.rand_like(x) < prob_to_keep).float()
            return (mask * x) / prob_to_keep
        else:
            return x