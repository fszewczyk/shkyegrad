from shkyegrad import Tensor

import numpy as np
from numpy.typing import NDArray

def get_uniform(shape: tuple[int, ...], min: float, max: float) -> NDArray:
    return np.random.uniform(low=min, high=max, size=shape)

def get_normal(shape: tuple[int, ...], mean: float, stdev: float) -> NDArray:
    return np.random.normal(loc=mean, scale=stdev, size=shape)

