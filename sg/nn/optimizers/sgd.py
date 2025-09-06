from shkyegrad import Tensor
from shkyegrad.sg.nn import Optimizer

import numpy as np

class SGD(Optimizer):
    def __init__(self, parameters, lr: float, momentum: float = 0):
        super().__init__(parameters)
        
        self.__acc_gradients = [Tensor(np.zeros_like(param.grad.data), requires_grad=False) for param in self._parameters]
        self.lr = lr
        self.momentum = momentum

    def step(self):
        for param, acc_grad in zip(self._parameters, self.__acc_gradients):
            acc_grad = self.momentum * acc_grad + (1 - self.momentum) * param.grad
        
        for param, acc_grad in zip(self._parameters, self.__acc_gradients):
            param -= acc_grad
