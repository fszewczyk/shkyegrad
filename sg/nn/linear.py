from shkyegrad.sg.nn.module import Module
import shkyegrad.sg.nn.utils as utils
from shkyegrad import Tensor

import numpy as np

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, requires_grad=True, dtype=np.float32):
        self.weights = Tensor(utils.get_normal((in_dim, out_dim), 0, 1).astype(dtype), requires_grad=requires_grad)
        self.bias = Tensor(utils.get_normal((out_dim,), 0, 1).astype(dtype), requires_grad=requires_grad)

    def forward(self, x) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x @ self.weights + self.bias
