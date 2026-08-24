"""
Post-hoc evaluation metrics on numpy masks (for test.py, computed once per predicted/ground-truth
mask pair — distinct from promisenet.losses.DiceIoU, which operates on batched torch tensors during
training/validation).

`hd95` is a direct Python port of the source repo's MATLAB evaluation pipeline
(Experiments_2.0/Evaluation/computeHD95.m): extract each mask's boundary pixels, compute the
symmetric nearest-neighbor distances (GT->pred and pred->GT), and take the 95th percentile of the
combined set. Requires scipy (ndimage for boundary extraction via binary erosion, spatial.cKDTree
for nearest-neighbor lookup).
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Binary mask -> boundary mask (mask minus its erosion), matching MATLAB's bwperim."""
    mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~eroded


def hd95(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance between two binary masks, in pixels.
    Returns NaN if either mask's boundary is empty (matches the MATLAB script's behavior)."""
    pred_edges = np.argwhere(_boundary(pred_mask))
    true_edges = np.argwhere(_boundary(true_mask))

    if pred_edges.size == 0 or true_edges.size == 0:
        return float("nan")

    tree_pred = cKDTree(pred_edges)
    tree_true = cKDTree(true_edges)

    d_gt_to_pred, _ = tree_pred.query(true_edges)   # GT -> Pred
    d_pred_to_gt, _ = tree_true.query(pred_edges)   # Pred -> GT

    return float(np.percentile(np.concatenate([d_gt_to_pred, d_pred_to_gt]), 95))


def dice_iou(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-5):
    """Binary Dice and IoU for a single class, on numpy masks."""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.sum(pred_mask & true_mask)
    total = np.sum(pred_mask) + np.sum(true_mask)
    union = total - intersection
    dice = (2. * intersection + smooth) / (total + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return float(dice), float(iou)


def fpr_fnr(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-5):
    """Binary False Positive / False Negative Rate for a single class, on numpy masks."""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    TP = np.sum(pred_mask & true_mask)
    FP = np.sum(pred_mask & ~true_mask)
    FN = np.sum(~pred_mask & true_mask)
    TN = np.sum(~pred_mask & ~true_mask)
    fpr = (FP + smooth) / (FP + TN + smooth)
    fnr = (FN + smooth) / (FN + TP + smooth)
    return float(fpr), float(fnr)


def evaluate_binary_mask(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """All metrics for one binary prediction/ground-truth mask pair."""
    dice, iou = dice_iou(pred_mask, true_mask)
    fpr, fnr = fpr_fnr(pred_mask, true_mask)
    return {"DICE": dice, "IoU": iou, "HD95": hd95(pred_mask, true_mask), "FNR": fnr, "FPR": fpr}


def evaluate_multiclass_mask(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int) -> dict:
    """All metrics for one multiclass prediction/ground-truth mask pair (e.g. CAMUS), per class."""
    out = {}
    for c in range(num_classes):
        m = evaluate_binary_mask(pred_mask == c, true_mask == c)
        for k, v in m.items():
            out[f"{k}_C{c}"] = v
    return out
