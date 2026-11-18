import torch

@torch.no_grad()
def batch_metrics(logits, targets, threshold=.5, eps=1e-7):
    pred = torch.sigmoid(logits) >= threshold
    truth = targets >= .5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    tn = (~pred & ~truth).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_dice": dice, "iou": iou}

def mean_metrics(rows):
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}
