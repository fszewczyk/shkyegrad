import torch

def MSE(pred, target):
    return torch.mean(torch.square(pred - target))

# to be applied after sigmoid
def BCELoss(pred, target, eps=1e-7):
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()

# to be applied on raw logits
# target is one-hot encodings
def CrossEntropyLoss(pred, target, eps=1e-7):
    max_logits, _ = pred.max(dim=1, keepdim=True)
    centered_logits = pred - max_logits
    exp_logits = torch.exp(centered_logits)
    sum_exp = exp_logits.sum(dim=1, keepdim=True)
    log_probs = centered_logits - torch.log(sum_exp)
    
    target_indices = target.argmax(dim=1)
    loss = -log_probs[range(pred.shape[0]), target_indices].mean()
    return loss