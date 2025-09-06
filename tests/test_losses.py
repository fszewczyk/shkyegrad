from shkyegrad import Tensor
import shkyegrad.losses as L

import numpy as np

def test_mse_loss():
    pred = Tensor(np.array([0.9, 0.1]), requires_grad=True)
    target = Tensor(np.array([1.0, 0.0]))
    loss = L.mse(pred, target)
    assert np.allclose(loss.data, 0.01)

    loss.backward()
    expected_grad = 2 * (pred.data - target.data) / pred.data.size
    assert np.allclose(pred.grad, expected_grad)

def test_mae_forward_and_backward():
    pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    target = Tensor(np.array([2.0, 2.0, 2.0]))

    loss = L.mae(pred, target)
    expected_loss = np.mean(np.abs(pred.data - target.data))
    assert np.allclose(loss.data, expected_loss)

    loss.backward()
    expected_grad = np.sign(pred.data - target.data) / pred.data.size
    assert np.allclose(pred.grad, expected_grad)

