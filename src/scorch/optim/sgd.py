import torch
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.__velocities = [torch.zeros_like(p.data) for p in self._params]

    def step(self):
        for p, v in zip(self._params, self.__velocities):
            if p.grad is not None:
                v = self.momentum * v + p.grad
                p.data -= self.lr * v
