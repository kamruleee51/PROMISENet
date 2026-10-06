"""
Evaluate a trained PROMISE-CNN / PROMISE-Txformer checkpoint on a test split, writing per-image
metrics (DSC/IoU/HD95/FNR/FPR) to a CSV and predicted masks to disk.

Usage:
    python test.py --config configs/isic2017.yaml --model cnn --data_dir ./data/isic2017 \
                    --weights outputs/isic2017_cnn_best.pth --out_dir outputs/isic2017_cnn_test
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from promisenet.models.promisenet_cnn import PROMISENet_CNN
from promisenet.models.promisenet_txformer import PROMISENet_txformer
from promisenet.data.datasets import ISICDataset, CAMUSDataset, KvasirDataset, custom_collate_fn
from promisenet.metrics import evaluate_binary_mask, evaluate_multiclass_mask


def build_test_dataset(cfg, data_dir, model_key):
    dataset_type = cfg["dataset_type"]
    img_size = cfg.get("img_size", 256)

    if dataset_type == "isic2017":
        random_perturb = cfg[model_key]["random_bbox_perturb"]
        return ISICDataset(
            sorted(glob(os.path.join(data_dir, "test_images", "*.jpg"))),
            sorted(glob(os.path.join(data_dir, "test_masks", "*.png"))),
            img_size=img_size, random_perturb=random_perturb,
        )
    if dataset_type == "camus":
        return CAMUSDataset(
            sorted(glob(os.path.join(data_dir, "test", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "test", "masks", "*", "*.png"))),
        )
    if dataset_type in ("kvasir_polyp", "kvasir_instrument"):
        return KvasirDataset(
            sorted(glob(os.path.join(data_dir, "test", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "test", "masks", "*", "*.png"))),
        )
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_model(cfg, model):
    num_classes = cfg["num_classes"]
    in_ch = cfg.get("num_input_channels", 3)
    img_size = cfg.get("img_size", 256)
    if model == "cnn":
        return PROMISENet_CNN(num_input_channels=in_ch, num_classes=num_classes, image_size=(img_size, img_size))
    return PROMISENet_txformer(in_channels=in_ch, out_channels=num_classes, img_size=(img_size, img_size))


def main():
    parser = argparse.ArgumentParser(description="Test PROMISENet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "transformer"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to a _best.pth checkpoint")
    parser.add_argument("--out_dir", type=str, default="./outputs/test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_key = "cnn" if args.model == "cnn" else "txformer"
    num_classes = cfg["num_classes"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "predictions"), exist_ok=True)

    test_dataset = build_test_dataset(cfg, args.data_dir, model_key)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    print(f"Test samples: {len(test_dataset)}")

    model = build_model(cfg, args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)
            name = batch["name"][0]

            predictions = model(images, bbox)
            # PROMISE-CNN already returns softmax probabilities; PROMISE-Txformer returns raw
            # logits — argmax is invariant to softmax, so no conversion is needed here either way.
            pred_mask = torch.argmax(predictions, dim=1)[0].cpu().numpy().astype(np.uint8)
            true_mask = masks[0, 0].cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(args.out_dir, "predictions", os.path.basename(name) + "_predicted.png"), pred_mask)

            if num_classes == 2:
                row = evaluate_binary_mask(pred_mask, true_mask)
            else:
                row = evaluate_multiclass_mask(pred_mask, true_mask, num_classes)
            row["name"] = os.path.basename(name)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    numeric_cols = [c for c in df.columns if c != "name"]
    summary = df[numeric_cols].agg(["mean", "std"])
    print(summary)
    summary.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"))
    print(f"Saved per-image metrics and predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
