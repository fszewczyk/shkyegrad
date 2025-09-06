import torch

def MSE(pred, target):
    return torch.mean(torch.square(pred - target))


def BCELoss(pred, target, eps=1e-7):
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()