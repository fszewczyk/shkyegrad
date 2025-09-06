from shkyegrad import Tensor

import numpy as np

class Optimizer:
    def __init__(self, parameters: list[Tensor]):
        self._parameters = parameters

    def zero_grad(self):
        for param in self._parameters:
            param.grad = np.zeros_like(param.data)

