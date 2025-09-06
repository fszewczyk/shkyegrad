from shkyegrad import Tensor
import numpy as np

def mse(pred: Tensor, target: Tensor) -> Tensor:
    se = (pred - target) ** 2
    return se.mean()

def mae(pred: Tensor, target: Tensor) -> Tensor:
    ae = (pred - target).abs()
    return ae.mean()


