import torch

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0)

class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

class Softmax(torch.nn.Module):
    def __init__(self, dim: int = -1, temperature: float = 1):
        super().__init__()

        self.t = temperature
        self.dim = dim

    def forward(self, logits):
        scaled_logits = logits / self.t
        max_vals, _ = torch.max(logits, dim=self.dim, keepdim=True)
        shifted_logits = scaled_logits - max_vals
        exp_logits = torch.exp(shifted_logits)
        return exp_logits / torch.sum(exp_logits, dim=self.dim, keepdim=True)

