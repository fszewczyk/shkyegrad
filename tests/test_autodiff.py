import numpy as np
from autograd import grad as ag_grad
import autograd.numpy as anp

from shkyegrad import Tensor


def test_large_computation_graph():
    x_np = np.random.randn(3, 4)
    y_np = np.random.randn(4, 5)

    x = Tensor(x_np.copy(), requires_grad=True)
    y = Tensor(y_np.copy(), requires_grad=True)

    z = (x @ y).tanh()
    loss = z.sum()
    loss.backward()

    grad_x = x.grad
    grad_y = y.grad

    def f(x_val, y_val):
        z = anp.tanh(anp.matmul(x_val, y_val))
        return anp.sum(z)

    grad_f_x = ag_grad(f, 0)(x_np, y_np)
    grad_f_y = ag_grad(f, 1)(x_np, y_np)

    assert np.allclose(grad_x, grad_f_x, atol=1e-5), "x.grad mismatch"
    assert np.allclose(grad_y, grad_f_y, atol=1e-5), "y.grad mismatch"
