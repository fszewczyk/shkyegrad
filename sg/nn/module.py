from shkyegrad import Tensor

import numpy as np
from collections.abc import Iterable

class Module:
    def parameters(self):
        params = []
        for member in self.__dict__.values():
            if isinstance(member, Module):
                params.extend(member.parameters())
            elif isinstance(member, Tensor) and member.requires_grad:
                params.append(member)
            elif isinstance(member, Iterable):
                for member_in_iter in member:
                    if isinstance(member, Module):
                        params.extend(member.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



