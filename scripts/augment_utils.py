import os
import cv2
import random
import albumentations as A
from tqdm import tqdm


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, subject):
        image, mask = subject['image'], subject['mask']
        aug = A.Compose([
            A.HorizontalFlip(p=self.p),
            A.VerticalFlip(p=self.p),
            A.Rotate(limit=360, border_mode=0, p=self.p)
        ], p=self.p)
        augmented = aug(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        return image, mask


class HFlip:
    def __init__(self, horizontal=True, p=0.5):
        self.horizontal = horizontal
        self.p = p

    def __call__(self, subject):
        image, mask = subject['image'], subject['mask']
        trans = A.HorizontalFlip(p=self.p)
        augmented = trans(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        return image, mask


class VFlip:
    def __init__(self, vertical=True, p=0.5):
        self.vertical = vertical
        self.p = p

    def __call__(self, subject):
        image, mask = subject['image'], subject['mask']
        trans = A.VerticalFlip(p=self.p)
        augmented = trans(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        return image, mask


class Rotate:
    def __init__(self, rotate=True, p=0.5):
        self.rotate = rotate
        self.p = p

    def __call__(self, subject):
        image, mask = subject['image'], subject['mask']
        angle = random.uniform(0, 360)
        trans = A.Rotate(limit=(angle, angle), border_mode=0, p=self.p)
        augmented = trans(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        return image, mask


def read_img(path):
    img_ = cv2.imread(path, 1)
    img_ = cv2.resize(img_, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img_


def read_mask(path):
    mask_ = cv2.imread(path, 0)
    mask_ = cv2.resize(mask_, (256, 256), interpolation=cv2.INTER_NEAREST)
    return mask_


def augment_split(im_dir, mask_dir, out_base_img, out_base_mask):
    """
    Augment a single split (train/val/test) by producing:
    original, H_flipped, V_flipped, rotated, combined
    into subdirectories under out_base_img and out_base_mask.
    """
    aug_types = ['original', 'H_flipped', 'V_flipped', 'rotated', 'combined']
    for aug_name in aug_types:
        os.makedirs(os.path.join(out_base_img, aug_name), exist_ok=True)
        os.makedirs(os.path.join(out_base_mask, aug_name), exist_ok=True)

    C = RandomFlip(p=0.8)
    H = HFlip(p=1)
    V = VFlip(p=1)
    R = Rotate(p=1)

    files = [f for f in os.listdir(im_dir) if f.endswith('.png')]
    print(f"  Augmenting {len(files)} images...")

    for fname in tqdm(files):
        im = read_img(os.path.join(im_dir, fname))
        mask = read_mask(os.path.join(mask_dir, fname))

        # Original
        cv2.imwrite(os.path.join(out_base_img, 'original', fname), im)
        cv2.imwrite(os.path.join(out_base_mask, 'original', fname), mask)

        # Combined (random flip + rotate)
        im_c, mask_c = C({'image': im, 'mask': mask})
        cv2.imwrite(os.path.join(out_base_img, 'combined', fname), im_c)
        cv2.imwrite(os.path.join(out_base_mask, 'combined', fname), mask_c)

        # Horizontal flip
        im_h, mask_h = H({'image': im, 'mask': mask})
        cv2.imwrite(os.path.join(out_base_img, 'H_flipped', fname), im_h)
        cv2.imwrite(os.path.join(out_base_mask, 'H_flipped', fname), mask_h)

        # Vertical flip
        im_v, mask_v = V({'image': im, 'mask': mask})
        cv2.imwrite(os.path.join(out_base_img, 'V_flipped', fname), im_v)
        cv2.imwrite(os.path.join(out_base_mask, 'V_flipped', fname), mask_v)

        # Rotation
        im_r, mask_r = R({'image': im, 'mask': mask})
        cv2.imwrite(os.path.join(out_base_img, 'rotated', fname), im_r)
        cv2.imwrite(os.path.join(out_base_mask, 'rotated', fname), mask_r)
