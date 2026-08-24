"""
Dataset classes for the four benchmarks.

ISICDataset: single-box binary dataset. `random_perturb` must be set per the model being trained —
    False for PROMISE-CNN, True for PROMISE-Txformer (see bbox_utils.get_bounding_box's docstring for
    why these differ; it's a real discrepancy between the two source notebooks, preserved here).

CAMUSDataset: multi-class, multi-box (one box per cardiac chamber) dataset.

KvasirDataset: shared by Kvasir-Polyp and Kvasir-Instrument — both binary, both use contour-based
    multi-box extraction (typically yielding one box, since these are usually single-blob masks),
    identical between PROMISE-CNN and PROMISE-Txformer (no random_perturb split needed here, unlike
    ISIC).

All three need `custom_collate_fn` for batching, since the number of boxes per image can vary (CAMUS:
different numbers of visible chambers; Kvasir: occasionally zero or multiple polyp/instrument blobs).
ISICDataset always yields exactly one box, so it works with `custom_collate_fn` too (max_boxes=1).
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from promisenet.data.bbox_utils import get_bounding_box, get_bounding_boxes_from_mask


class ISICDataset(Dataset):
    """ISIC-2017 skin lesion dataset. No offline augmentation (per the paper's training protocol —
    ISIC-2017 is the one dataset that is NOT offline-augmented, unlike the other three)."""

    def __init__(self, image_paths, mask_paths, img_size: int = 256, random_perturb: bool = False, transform=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.img_size = img_size
        self.random_perturb = random_perturb
        self.transform = transform
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in number of images and masks."

    def __len__(self):
        return len(self.image_paths)

    def _read_img(self, path):
        img_ = cv2.imread(path, 1)
        img_ = cv2.resize(img_, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        return img_ / img_.max()

    def _read_mask(self, path):
        mask_ = cv2.imread(path, 0)
        mask_ = cv2.resize(mask_, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_[mask_ == 0] = 0
        mask_[mask_ == 255] = 1
        return mask_

    def __getitem__(self, idx):
        image_head, tail = os.path.split(self.image_paths[idx])
        mask_head, _ = os.path.split(self.mask_paths[idx])

        image = self._read_img(os.path.join(image_head, tail))
        mask = self._read_mask(os.path.join(mask_head, tail.rsplit(".", 1)[0] + "_segmentation.png"))
        bbox = get_bounding_box(mask, random_perturb=self.random_perturb, max_perturb=20)

        combined = {"image": image, "mask": mask, "bbox": [bbox], "name": tail}

        if self.transform:
            combined = self.transform(combined)

        combined["image"] = torch.tensor(combined["image"], dtype=torch.float32).permute(2, 0, 1)
        combined["mask"] = torch.tensor(combined["mask"], dtype=torch.float32).unsqueeze(0)
        combined["bbox"] = torch.tensor(combined["bbox"], dtype=torch.float32)  # [1, 4]
        return combined


class CAMUSDataset(Dataset):
    """CAMUS 2D echocardiography dataset (multi-class: background + LV + MYO + LA).
    Expects pre-augmented data from scripts/prepare_camus.py."""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_path = sorted(image_paths)
        self.mask_path = sorted(mask_paths)
        self.transform = transform
        assert len(self.image_path) == len(self.mask_path), "Mismatch in number of images and masks."

    def __len__(self):
        return len(self.image_path)

    def _read_img(self, path):
        img_ = cv2.imread(path, 1)
        return img_ / img_.max()

    def _read_mask(self, path):
        mask_ = cv2.imread(path, 0)
        return cv2.medianBlur(mask_, 7)

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        mask_name = self.mask_path[idx]

        image = self._read_img(image_name)
        mask = self._read_mask(mask_name)

        mask_binary = mask.copy()
        mask_binary[mask_binary != 0] = 1
        box_list = get_bounding_boxes_from_mask(mask_binary, max_perturb=10)

        combined = {"image": image, "mask": mask, "bbox": box_list, "name": image_name}

        if self.transform:
            trans = self.transform(image=combined["image"], mask=combined["mask"])
            combined["image"], combined["mask"] = trans["image"], trans["mask"]

        combined["image"] = torch.tensor(np.array(combined["image"]), dtype=torch.float32).permute(2, 0, 1)
        combined["mask"] = torch.tensor(np.array(combined["mask"]), dtype=torch.float32).unsqueeze(0)
        combined["bbox"] = torch.tensor(combined["bbox"], dtype=torch.float32) if combined["bbox"] else torch.zeros((1, 4))
        return combined


class KvasirDataset(Dataset):
    """Shared by Kvasir-Polyp and Kvasir-Instrument (both binary, contour-based bbox extraction).
    Both are offline-augmented per the paper's protocol — expects pre-augmented data from
    scripts/prepare_kvasir.py (see docs/DATASETS.md)."""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_path = sorted(image_paths)
        self.mask_path = sorted(mask_paths)
        self.transform = transform
        assert len(self.image_path) == len(self.mask_path), "Mismatch in number of images and masks."

    def __len__(self):
        return len(self.image_path)

    def _read_img(self, path):
        img_ = cv2.imread(path, 1)
        return img_ / img_.max()

    def _read_mask(self, path):
        mask_ = cv2.imread(path, 0)
        mask_[mask_ == 0] = 0
        mask_[mask_ != 0] = 1
        return cv2.medianBlur(mask_, 7)

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        mask_name = self.mask_path[idx]

        image = self._read_img(image_name)
        mask = self._read_mask(mask_name)
        box_list = get_bounding_boxes_from_mask(mask, max_perturb=10)

        combined = {"image": image, "mask": mask, "bbox": box_list, "name": image_name}

        if self.transform:
            trans = self.transform(image=combined["image"], mask=combined["mask"])
            combined["image"], combined["mask"] = trans["image"], trans["mask"]

        combined["image"] = torch.tensor(np.array(combined["image"]), dtype=torch.float32).permute(2, 0, 1)
        combined["mask"] = torch.tensor(np.array(combined["mask"]), dtype=torch.float32).unsqueeze(0)
        combined["bbox"] = torch.tensor(combined["bbox"], dtype=torch.float32) if combined["bbox"] else torch.zeros((1, 4))
        return combined


def custom_collate_fn(batch):
    """Pads variable-length bounding-box lists to [B, max_N, 4] within a batch (zero-padding, which
    BoxPositionEncoder treats as "no box here" and skips)."""
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    names = [item["name"] for item in batch]

    max_boxes = max(len(item["bbox"]) for item in batch)
    padded_bboxes = []
    for item in batch:
        bbox = item["bbox"]
        if len(bbox) < max_boxes:
            padding = torch.zeros((max_boxes - len(bbox), 4), dtype=torch.float32)
            bbox = torch.cat((bbox, padding), dim=0)
        padded_bboxes.append(bbox)

    return {
        "image": images,
        "mask": masks,
        "bbox": torch.stack(padded_bboxes),  # [B, max_N, 4]
        "name": names,
    }
