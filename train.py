"""
Train PROMISE-CNN or PROMISE-Txformer on one of the four datasets.

Usage:
    python train.py --config configs/isic2017.yaml --model cnn --data_dir ./data/isic2017
    python train.py --config configs/camus.yaml --model transformer --data_dir ./data/camus_augmented
"""
import argparse
import os
from glob import glob

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from promisenet.models.promisenet_cnn import PROMISENet_CNN
from promisenet.models.promisenet_txformer import PROMISENet_txformer
from promisenet.data.datasets import ISICDataset, CAMUSDataset, KvasirDataset, custom_collate_fn
from promisenet.losses import set_seed, compute_loss


def build_datasets(cfg, data_dir, model_key):
    dataset_type = cfg["dataset_type"]
    img_size = cfg.get("img_size", 256)

    if dataset_type == "isic2017":
        random_perturb = cfg[model_key]["random_bbox_perturb"]
        train_ds = ISICDataset(
            sorted(glob(os.path.join(data_dir, "training_images", "*.jpg"))),
            sorted(glob(os.path.join(data_dir, "training_masks", "*.png"))),
            img_size=img_size, random_perturb=random_perturb,
        )
        val_ds = ISICDataset(
            sorted(glob(os.path.join(data_dir, "val_images", "*.jpg"))),
            sorted(glob(os.path.join(data_dir, "val_masks", "*.png"))),
            img_size=img_size, random_perturb=random_perturb,
        )
    elif dataset_type == "camus":
        train_ds = CAMUSDataset(
            sorted(glob(os.path.join(data_dir, "train", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "train", "masks", "*", "*.png"))),
        )
        val_ds = CAMUSDataset(
            sorted(glob(os.path.join(data_dir, "val", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "val", "masks", "*", "*.png"))),
        )
    elif dataset_type in ("kvasir_polyp", "kvasir_instrument"):
        train_ds = KvasirDataset(
            sorted(glob(os.path.join(data_dir, "train", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "train", "masks", "*", "*.png"))),
        )
        val_ds = KvasirDataset(
            sorted(glob(os.path.join(data_dir, "val", "images", "*", "*.png"))),
            sorted(glob(os.path.join(data_dir, "val", "masks", "*", "*.png"))),
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return train_ds, val_ds


def build_model(cfg, model):
    num_classes = cfg["num_classes"]
    in_ch = cfg.get("num_input_channels", 3)
    img_size = cfg.get("img_size", 256)
    if model == "cnn":
        return PROMISENet_CNN(num_input_channels=in_ch, num_classes=num_classes, image_size=(img_size, img_size))
    # PROMISE-Txformer's transformer hyperparameters (patch_size/hidden_dim/num_layers/num_heads/
    # mlp_dim/dropout_rate) are identical across all four datasets and use the class defaults —
    # only num_classes/num_channels/image_size vary.
    return PROMISENet_txformer(num_classes=num_classes, num_channels=in_ch, image_size=img_size)


def main():
    parser = argparse.ArgumentParser(description="Train PROMISENet")
    parser.add_argument("--config", type=str, required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--model", type=str, choices=["cnn", "transformer"], required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (see docs/DATASETS.md)")
    parser.add_argument("--epochs", type=int, default=None, help="Override config's epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch size")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_key = "cnn" if args.model == "cnn" else "txformer"
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    num_classes = cfg["num_classes"]
    dice_weight = cfg["dice_weight"]
    ce_weight = cfg["ce_weight"]
    differentiable_dice = cfg[model_key]["differentiable_dice"]

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {cfg['dataset_type']} | Model: PROMISE-{'CNN' if args.model=='cnn' else 'Txformer'} | Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    train_dataset, val_dataset = build_datasets(cfg, args.data_dir, model_key)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    model = build_model(cfg, args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    filename = f"{cfg['dataset_type']}_{args.model}"
    max_val_dice = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print("-" * 100)
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss, train_dice, n_batches = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc="train"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)

            optimizer.zero_grad()
            predictions = model(images, bbox)  # do NOT apply softmax here — see promisenet/losses.py
            loss, dice_scores, _ = compute_loss(
                predictions, masks, device, num_classes,
                dice_weight=dice_weight, ce_weight=ce_weight, differentiable_dice=differentiable_dice,
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += torch.mean(torch.stack(dice_scores[1:])).item()  # foreground classes only
            n_batches += 1

        print(f"Training Loss: {train_loss/n_batches:.4f} | Training Dice: {train_dice/n_batches:.4f}")

        model.eval()
        val_loss, val_dice, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                bbox = batch["bbox"].to(device)

                predictions = model(images, bbox)
                loss, dice_scores, _ = compute_loss(
                    predictions, masks, device, num_classes,
                    dice_weight=dice_weight, ce_weight=ce_weight, differentiable_dice=differentiable_dice,
                )
                val_loss += loss.item()
                val_dice += torch.mean(torch.stack(dice_scores[1:])).item()
                n_val += 1

        avg_val_dice = val_dice / n_val
        print(f"Validation Loss: {val_loss/n_val:.4f} | Validation Dice: {avg_val_dice:.4f}")

        if avg_val_dice > max_val_dice:
            max_val_dice = avg_val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.out_dir, filename + "_best.pth"))
            print(f"New best model saved (epoch {best_epoch})")

    torch.save(model.state_dict(), os.path.join(args.out_dir, filename + "_latest.pth"))
    print(f"Training complete. Best epoch: {best_epoch} (val Dice {max_val_dice:.4f})")


if __name__ == "__main__":
    main()
