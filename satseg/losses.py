import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()

def segmentation_loss(outputs, targets):
    if isinstance(outputs, dict):
        main = outputs["out"]
        loss = F.binary_cross_entropy_with_logits(main, targets) + dice_loss(main, targets)
        for aux in outputs.get("aux", []):
            loss = loss + 0.25 * (F.binary_cross_entropy_with_logits(aux, targets) + dice_loss(aux, targets))
        return loss
    return F.binary_cross_entropy_with_logits(outputs, targets) + dice_loss(outputs, targets)
