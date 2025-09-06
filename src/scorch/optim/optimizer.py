import abc

class Optimizer(abc.ABC):
    def __init__(self, params):
        self._params = list(params)

    def zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

    @abc.abstractmethod
    def step(self):
        pass

