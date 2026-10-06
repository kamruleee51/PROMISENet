"""
Prepare Kvasir-Polyp or Kvasir-Instrument with offline augmentations, using the same augmentation
pipeline as CAMUS (scripts/augment_utils.py's `augment_split`). Per the paper's training protocol,
offline augmentation is applied to CAMUS, Kvasir-Polyp, and Kvasir-Instrument (only ISIC-2017 is not).

Expects pre-separated data in:
    raw_dir/
        train/images/  train/masks/
        val/images/    val/masks/
        test/images/   test/masks/

Produces augmented data in:
    out_dir/
        train/images/{original,H_flipped,V_flipped,rotated,combined}/
        train/masks/{original,H_flipped,V_flipped,rotated,combined}/
        val/images/... val/masks/...
        test/images/... test/masks/...
"""
import argparse
import os
from augment_utils import augment_split


def main():
    parser = argparse.ArgumentParser(description="Prepare Kvasir-Polyp / Kvasir-Instrument dataset with offline augmentations")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to pre-separated dataset (train/val/test)")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save augmented dataset")
    parser.add_argument("--dataset_name", type=str, default="kvasir", choices=["kvasir_polyp", "kvasir_instrument"],
                         help="Only used for the progress messages below.")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        im_dir = os.path.join(args.raw_dir, split, "images")
        mask_dir = os.path.join(args.raw_dir, split, "masks")

        if not os.path.exists(im_dir) or not os.path.exists(mask_dir):
            print(f"Skipping {split} — paths not found: {im_dir}, {mask_dir}")
            continue

        out_img = os.path.join(args.out_dir, split, "images")
        out_mask = os.path.join(args.out_dir, split, "masks")

        print(f"Processing {args.dataset_name} {split} split...")
        augment_split(im_dir, mask_dir, out_img, out_mask)

    print(f"{args.dataset_name} augmentation complete!")


if __name__ == "__main__":
    main()
