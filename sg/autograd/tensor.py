import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union

class Tensor:
    def __init__(self,
                 data: Union[int, float, list[float], list[list[float]], NDArray],
                 dtype: Optional[np.dtype]=np.float32,
                 requires_grad: bool=False):
        self.data = np.asarray(data, data.dtype if hasattr(data, 'dtype') else dtype)
        self.grad: Optional[NDArray] = np.zeros_like(self.data) if requires_grad else None

        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._children: set["Tensor"] = set()
        self._op: str = ""

    def __reduce_grad(self, grad):
        while len(grad.shape) > len(self.data.shape):
            grad = grad.sum(axis=0)

        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, self.data.shape)):
            if t_dim == 1 and g_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, requires_grad={self.requires_grad}, data={self.data})"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> "Tensor":
        data = self.data[idx]
        out = Tensor(data, requires_grad = self.requires_grad)
        out._op = "slice"
        out._children = (self,)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad

        out._backward = _backward
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad = self.requires_grad or other.requires_grad)
        out._op = "add"
        out._children = set((self, other))

        def _backward():
            if self.requires_grad:
                self.grad += self.__reduce_grad(out.grad)
            if other.requires_grad:
                other.grad += other.__reduce_grad(out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad = self.requires_grad or other.requires_grad)
        out._op = "add"
        out._children = set((self, other))

        def _backward():
            if self.requires_grad:
                self.grad += self.__reduce_grad(out.grad)
            if other.requires_grad:
                other.grad -= other.__reduce_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad = self.requires_grad or other.requires_grad)
        out._op = "mul"
        out._children = set((self, other))

        def _backward():
            if self.requires_grad:
                self.grad += self.__reduce_grad(other.data * out.grad)
            if other.requires_grad:
                other.grad += other.__reduce_grad(self.data * out.grad)

        self._backward = _backward
        return out

    def __rmul__(self, other): return self * other

    def __truediv__(self, other: Union["Tensor", int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad = self.requires_grad or other.requires_grad)
        out._op = "div"
        out._children = set((self, other))

        def _backward():
            if self.requires_grad:
                self.grad += self.__reduce_grad((1.0 / other.data) * out.grad)
            if other.requires_grad:
                other.grad += other.__reduce_grad((-self.data / (other.data ** 2)) * out.grad)

        out._backward = _backward
        return out

    def __rtruediv__(self, other): return Tensor(other) / self

    def __pow__(self, exponent: Union["Tensor", int, float]) -> "Tensor":
        exponent = exponent if isinstance(exponent, Tensor) else Tensor(exponent)
        out = Tensor(self.data ** exponent.data, requires_grad = self.requires_grad or exponent.requires_grad)
        out._op = "pow"
        out._children = set((self, exponent))

        def _backward():
            if self.requires_grad:
                self.grad += (exponent.data * self.data ** (exponent.data - 1)) * out.grad
            if exponent.requires_grad:
                self_mask = self.data > 0
                safe_log_base = np.where(self_mask, self.data, 1.0)
                grad_exp = (out.data * np.log(safe_log_base)) * out.grad
                grad_exp[~self_mask] = 0
                exponent.grad += grad_exp

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("Both operands must be 2D matrices for matmul.")

        out_data = np.matmul(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad)
        out._op = "matmul"
        out._children = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad = self.requires_grad)
        out._op = "neg"
        out._children = set((self,))

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), requires_grad = self.requires_grad)
        out._op = "tanh"
        out._children = set((self,))

        def _backward():
            if self.requires_grad:
                self.grad += (1 - (out.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        positive_mask = self.data > 0
        out = Tensor(self.data * positive_mask, requires_grad = self.requires_grad)
        out._op = "relu"
        out._children = (self,)

        def _backward():
            if self.requires_grad:
                self.grad += positive_mask.astype(self.data.dtype) * out.grad

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(self.data.sum(), requires_grad = self.requires_grad)
        out._op = "sum"
        out._children = (self,)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        out = Tensor(self.data.mean(), requires_grad = self.requires_grad)
        out._op = "mean"
        out._children = set((self,))

        def _backward():
            if self.requires_grad:
                self.grad += (np.ones_like(self.data) / self.data.size) * out.grad

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), requires_grad = self.requires_grad)
        out._op = "abs"
        out._children = set((self,))

        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out
        

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a Tensor that does not require grad")

        assert self.grad is not None, "Grad is None, even though grad is required!"

        self.grad = np.ones_like(self.grad)

        visited = set()
        topo = []

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

