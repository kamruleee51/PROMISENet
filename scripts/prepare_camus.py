import argparse
import os
from augment_utils import augment_split


def main():
    parser = argparse.ArgumentParser(description="Prepare CAMUS dataset with offline augmentations")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to pre-separated CAMUS dataset (train/val/test)")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save augmented dataset")
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        im_dir = os.path.join(args.raw_dir, split, "images")
        mask_dir = os.path.join(args.raw_dir, split, "masks")

        if not os.path.exists(im_dir) or not os.path.exists(mask_dir):
            print(f"Skipping {split} — paths not found: {im_dir}, {mask_dir}")
            continue

        out_img = os.path.join(args.out_dir, split, "images")
        out_mask = os.path.join(args.out_dir, split, "masks")

        print(f"Processing CAMUS {split} split...")
        augment_split(im_dir, mask_dir, out_img, out_mask)

    print("CAMUS augmentation complete!")


if __name__ == "__main__":
    main()
