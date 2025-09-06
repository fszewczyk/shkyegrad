from shkyegrad import Tensor

import numpy as np
import random

class DataLoader:
    def __init__(self, dataset, batch_size: int = 1, random: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random = random
        self.indices = list(range(len(dataset)))
        self.current = 0

    def __iter__(self):
        if self.random:
            random.shuffle(indices)

        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration
    
        batch_content = [self.dataset[idx] for idx in self.indices[self.current:self.current + self.batch_size]]

        sample_x = batch_content[0][0]
        sample_y = batch_content[0][1]

        x_shape = sample_x.data.shape 
        y_shape = sample_y.data.shape 

        batch_x_np = np.zeros((self.batch_size, *x_shape), dtype=sample_x.data.dtype)
        batch_y_np = np.zeros((self.batch_size, *y_shape), dtype=sample_y.data.dtype)

        for i, (x, y) in enumerate(batch_content):
            batch_x_np[i] = x.data
            batch_y_np[i] = y.data

        self.current += self.batch_size

        return Tensor(batch_x_np), Tensor(batch_y_np)

