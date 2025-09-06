import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        sqrt_k = math.sqrt(1 / in_features)
        self.weights = torch.nn.Parameter(torch.rand(in_features, out_features, dtype=dtype) * 2 * sqrt_k - sqrt_k)
        self.bias = torch.nn.Parameter(torch.rand(out_features, dtype=dtype) * 2 * sqrt_k - sqrt_k)

    def forward(self, x):
        return x @ self.weights + self.bias
