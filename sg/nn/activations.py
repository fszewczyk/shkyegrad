from shkyegrad.sg.nn import Module
from shkyegrad import Tensor

class ReLU(Module):
    def forward(self, x: Tensor):
        return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
