"""
Automated bounding-box extraction from ground-truth masks.

Two genuinely different extraction functions are used across the four datasets — preserved here
exactly as verified in the source notebooks rather than unified into one, since unifying them would
change the training distribution for at least one dataset/model combination relative to what actually
produced the paper's reported numbers:

- ISIC-2017 uses `get_bounding_box`: a single box from the whole (binary) mask via np.where.
  PROMISE-CNN's ISIC notebook expands it by a FIXED +/-20px on each side (no randomness).
  PROMISE-Txformer's ISIC notebook expands it by a RANDOM [0,20)px offset per side, re-drawn every
  call (so every training epoch sees a different perturbation for the same image).
  Use `random_perturb=False` / `True` respectively to match.

- Kvasir-Polyp, Kvasir-Instrument, and CAMUS all use `get_bounding_boxes_from_mask`: one box per
  connected component via cv2.findContours (so CAMUS gets one box per cardiac chamber; the two
  Kvasir datasets typically have one blob, so a single box), with a RANDOM [0, max_perturb)px offset
  per side, max_perturb=10 for all three (not 20). Identical across PROMISE-CNN and PROMISE-Txformer
  for these three datasets — no per-model variant needed here.
"""
import numpy as np
import cv2


def get_bounding_box(ground_truth_map: np.ndarray, random_perturb: bool = False, max_perturb: int = 20):
    """Single bounding box for a binary mask (ISIC-2017).

    Args:
        ground_truth_map: 2D binary mask, foreground > 0.
        random_perturb: if True, each side is expanded by a random [0, max_perturb) offset
            (matches PROMISE-Txformer's ISIC notebook); if False, each side is expanded by a fixed
            `max_perturb` (matches PROMISE-CNN's ISIC notebook, default max_perturb=20).
        max_perturb: perturbation range/amount in pixels.

    Returns:
        [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    H, W = ground_truth_map.shape

    def offset():
        return np.random.randint(0, max_perturb) if random_perturb else max_perturb

    x_min = max(0, x_min - offset())
    x_max = min(W, x_max + offset())
    y_min = max(0, y_min - offset())
    y_max = min(H, y_max + offset())
    return [x_min, y_min, x_max, y_max]


def get_bounding_boxes_from_mask(mask: np.ndarray, perturbation: bool = True, max_perturb: int = 10, min_area: int = 50):
    """One bounding box per connected component (CAMUS, Kvasir-Polyp, Kvasir-Instrument).

    Args:
        mask: 2D binary mask (foreground != 0). Internally rescaled to {0, 255} for cv2.findContours.
        perturbation: if True, each box side gets a random [0, max_perturb) offset.
        max_perturb: perturbation range in pixels (10 for all three datasets that use this function).
        min_area: connected components with contour area below this are dropped as noise.

    Returns:
        List of [x_min, y_min, x_max, y_max], one per component (possibly empty).
    """
    mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = mask.shape
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        if perturbation:
            x_min = max(0, x - np.random.randint(0, max_perturb))
            y_min = max(0, y - np.random.randint(0, max_perturb))
            x_max = min(W, x + w + np.random.randint(0, max_perturb))
            y_max = min(H, y + h + np.random.randint(0, max_perturb))
        else:
            x_min, y_min, x_max, y_max = x, y, x + w, y + h

        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes
