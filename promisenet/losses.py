import random

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_one_hot(labels: torch.Tensor, device, C: int = 2) -> torch.Tensor:
    """labels: [N, 1, H, W] integer class map -> [N, C, H, W] one-hot."""
    labels = labels.long().to(device)
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    return Variable(target)


class DiceIoU(nn.Module):
    """Per-class Dice and IoU, for metrics (not necessarily differentiable — depends on caller)."""

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, targets):
        dice_scores, iou_scores = [], []
        for class_idx in range(self.num_classes):
            p = predictions[:, class_idx, :, :]
            t = targets[:, class_idx, :, :]
            intersection = torch.sum(p * t)
            total_sum = torch.sum(p) + torch.sum(t)
            union = total_sum - intersection
            dice_scores.append((2. * intersection + self.smooth) / (total_sum + self.smooth))
            iou_scores.append((intersection + self.smooth) / (union + self.smooth))
        return dice_scores, iou_scores


class DiceLoss(nn.Module):
    """Dice loss averaged over foreground classes (background, class 0, is skipped)."""

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, targets):
        dice_losses = []
        for class_idx in range(1, self.num_classes):
            p = predictions[:, class_idx, :, :]
            t = targets[:, class_idx, :, :]
            intersection = torch.sum(p * t)
            union = torch.sum(p) + torch.sum(t)
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1. - dice_score)
        return torch.mean(torch.stack(dice_losses))


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    device,
    num_classes: int,
    dice_weight: float = 1.0,
    ce_weight: float = 0.5,
    differentiable_dice: bool = False,
):
    """
    Args:
        predictions: model output, passed through UNCHANGED from `model(...)` — softmax
            probabilities for PROMISE-CNN, raw logits for PROMISE-Txformer. [B, num_classes, H, W]
        targets: [B, 1, H, W] integer class map.
        dice_weight, ce_weight: see module docstring — 1.0/0.5 for binary datasets, 0.6/0.4 for CAMUS.
        differentiable_dice: see module docstring — False everywhere except PROMISE-Txformer on the
            three binary datasets.

    Returns:
        (total_loss, dice_scores, iou_scores) — dice_scores/iou_scores are always computed from the
        hard (argmax) prediction for reporting, regardless of `differentiable_dice`.
    """
    dice_loss_fn = DiceLoss(num_classes=num_classes)
    dice_iou_fn = DiceIoU(num_classes=num_classes)
    cross_entropy_loss = nn.CrossEntropyLoss()

    targets_one_hot = make_one_hot(targets, device, C=num_classes)
    cross_entropy = cross_entropy_loss(predictions, targets.squeeze(1).long())

    if differentiable_dice:
        predictions_softmax = torch.softmax(predictions, dim=1)
        dice_loss = dice_loss_fn(predictions_softmax, targets_one_hot)
        with torch.no_grad():
            segmented = torch.argmax(predictions, dim=1, keepdim=True)
            dice_scores, iou_scores = dice_iou_fn(make_one_hot(segmented, device, C=num_classes), targets_one_hot)
    else:
        segmented = torch.argmax(predictions, dim=1, keepdim=True).float()
        segmented_one_hot = make_one_hot(segmented, device, C=num_classes)
        dice_loss = dice_loss_fn(targets_one_hot, segmented_one_hot)
        dice_scores, iou_scores = dice_iou_fn(targets_one_hot, segmented_one_hot)

    total_loss = dice_weight * dice_loss + ce_weight * cross_entropy
    return total_loss, dice_scores, iou_scores


def compute_fpr_fnr(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-5):
    """Per-class False Positive / False Negative Rate."""
    pred_classes = torch.argmax(predictions, dim=1)
    target_classes = targets.squeeze(1)

    fpr_list, fnr_list = [], []
    for cls in range(num_classes):
        pred_mask = pred_classes == cls
        target_mask = target_classes == cls

        TP = torch.sum(pred_mask & target_mask).float()
        FP = torch.sum(pred_mask & ~target_mask).float()
        FN = torch.sum(~pred_mask & target_mask).float()
        TN = torch.sum(~pred_mask & ~target_mask).float()

        fpr_list.append(((FP + smooth) / (FP + TN + smooth)).item())
        fnr_list.append(((FN + smooth) / (FN + TP + smooth)).item())

    avg_fpr = sum(fpr_list[1:]) / (num_classes - 1) if num_classes > 1 else fpr_list[0]
    avg_fnr = sum(fnr_list[1:]) / (num_classes - 1) if num_classes > 1 else fnr_list[0]
    return fpr_list, fnr_list, avg_fpr, avg_fnr
