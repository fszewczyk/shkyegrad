import numpy as np
import pytest
from shkyegrad import Tensor


def test_tensor_creation_default_dtype():
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == np.float32
    assert not t.requires_grad


def test_tensor_creation_custom_dtype():
    t = Tensor([1, 2, 3], dtype=np.float64, requires_grad=True)
    assert t.data.dtype == np.float64
    assert t.requires_grad

def test_tensor_addition():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    assert np.allclose(c.data, [5, 7, 9])
    assert isinstance(c, Tensor)


def test_tensor_subtraction():
    a = Tensor([10, 20, 30])
    b = Tensor([1, 2, 3])
    c = a - b
    assert np.allclose(c.data, [9, 18, 27])
    assert isinstance(c, Tensor)


def test_add_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    c.backward()

    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0, 1.0])


def test_sub_backward():
    a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    c = a - b
    c.backward()

    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    assert np.allclose(b.grad, [-1.0, -1.0, -1.0])


def test_add_sub_chain_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    d = c - a
    d.backward()

    assert np.allclose(a.grad, [0.0, 0.0, 0.0])
    assert np.allclose(b.grad, [1.0, 1.0, 1.0])


def test_tensor_scalar_mul_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 3.0
    y.backward()

    assert np.allclose(y.data, [3, 6, 9])
    assert np.allclose(x.grad, [3.0, 3.0, 3.0])


def test_tensor_scalar_div_backward():
    x = Tensor([2.0, 4.0, 8.0], requires_grad=True)
    y = x / 2.0
    y.backward()

    assert np.allclose(y.data, [1, 2, 4])
    assert np.allclose(x.grad, [0.5, 0.5, 0.5])


def test_scalar_div_tensor_backward():
    x = Tensor([2.0, 4.0, 8.0], requires_grad=True)
    y = 16.0 / x
    y.backward()

    assert np.allclose(y.data, [8, 4, 2])
    assert np.allclose(x.grad, -16.0 / (x.data ** 2))


def test_elementwise_vector_multiplication():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a * b
    c.backward()

    assert np.allclose(c.data, [4, 10, 18])
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


def test_elementwise_matrix_multiplication():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    c = a * b
    c.backward()

    assert np.allclose(c.data, [[10, 40], [90, 160]])
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)

def test_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    c = a @ b
    assert np.allclose(c.data, a.data @ b.data)

    c.backward()

    expected_a_grad = np.ones_like(c.data) @ b.data.T
    expected_b_grad = a.data.T @ np.ones_like(c.data)

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)

def test_batched_matmul():
    a_data = np.random.randn(4, 2, 3)
    b_data = np.random.randn(4, 3, 5)

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    c = a @ b
    expected = np.matmul(a.data, b.data)
    assert np.allclose(c.data, expected)

    c.backward()

    expected_a_grad = np.matmul(np.ones_like(c.data), np.swapaxes(b_data, -1, -2))
    expected_b_grad = np.matmul(np.swapaxes(a_data, -1, -2), np.ones_like(c.data))

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)


def test_matmul_broadcasting():
    A = Tensor(np.random.randn(4, 2, 3), requires_grad=True)
    B = Tensor(np.random.randn(3, 5), requires_grad=True)
    C = A @ B
    assert np.allclose(C.data, A.data @ B.data)

    C.backward()

    expected_A_grad = np.matmul(np.ones_like(C.data), B.data.T)
    expected_B_grad = np.sum(np.matmul(np.transpose(A.data, (0, 2, 1)), np.ones_like(C.data)), axis=0)

    assert np.allclose(A.grad, expected_A_grad)
    assert np.allclose(B.grad, expected_B_grad)


def test_sum():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    y = x.sum()
    y.backward()
    assert y.data == 10.0
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_multiply_and_sum():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    y = Tensor(2.0, requires_grad=True)
    z = x * y
    w = z.sum()
    assert w.data == 20.0

    w.backward()
    assert np.allclose(x.grad, 2.0 * np.ones_like(x.data))

def test_neg_backward():
    x = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    y = -x
    z = y.sum()
    assert np.allclose(z.data, 1)

    z.backward()
    assert np.allclose(x.grad, -np.ones_like(x.data))

def test_tanh_backward():
    x_data = np.array([[0.0, 1.0], [-1.0, 2.0]])
    x = Tensor(x_data, requires_grad=True)
    y = x.tanh()
    y.sum().backward()

    expected_grad = 1 - np.tanh(x_data) ** 2
    assert np.allclose(x.grad, expected_grad)

def test_pow_scalar():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y = x ** 2.0
    assert np.allclose(y.data, [4, 9])

    y.sum().backward()
    assert np.allclose(x.grad, [4.0, 6.0])

def test_pow_tensor():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    e = Tensor(np.array([3.0, 2.0]), requires_grad=True)
    y = x ** e
    assert np.allclose(y.data, [8, 9])

    y.backward()

    expected_x = e.data * x.data ** (e.data - 1)
    expected_e = (x.data ** e.data) * np.log(x.data)
    assert np.allclose(x.grad, expected_x)
    assert np.allclose(e.grad, expected_e)

def test_relu_backward():
    x = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
    y = x.relu()
    assert np.allclose(y.data, [0, 0, 2])

    y.backward()
    expected_grad = np.array([0.0, 0.0, 1.0])
    assert np.allclose(x.grad, expected_grad)
