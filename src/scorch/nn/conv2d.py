import torch
import math

class Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        kW, kH = kernel_size
        weight_shape = (out_channels, in_channels, kH, kW)
        sqrt_k = math.sqrt(1.0 / (in_channels * kW * kH))
        self.weights = torch.nn.Parameter(torch.rand(weight_shape) * 2.0 * sqrt_k - sqrt_k)
        self.bias = torch.nn.Parameter(torch.rand((out_channels,)) * 2.0 * sqrt_k - sqrt_k)

    def forward(self, x):
        return torch.nn.functional.conv2d(
                x,
                self.weights,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding)

