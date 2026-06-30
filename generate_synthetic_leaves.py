"""
generate_synthetic_leaves.py
Generates a small synthetic leaf-image dataset in ImageFolder format, purely
to let the training/inference pipeline run end-to-end without a manual
dataset download.

SWAP-IN INSTRUCTIONS for the real dataset:
1. Download "New Plant Diseases Dataset" or "PlantVillage" from Kaggle.
2. Arrange it as:
       data/train/<class_name>/*.jpg
       data/val/<class_name>/*.jpg
3. No code changes needed elsewhere — train_model.py uses torchvision's
   ImageFolder, which reads class names directly from folder names.

Usage:
    python generate_synthetic_leaves.py --out ../data --n_per_class 120
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["healthy", "leaf_blight", "leaf_rust"]
IMG_SIZE = 224


def make_leaf_base(rng) -> Image.Image:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    green = (
        int(rng.integers(40, 90)),
        int(rng.integers(110, 160)),
        int(rng.integers(30, 70)),
    )
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    draw.ellipse([cx - 90, cy - 60, cx + 90, cy + 60], fill=green)
    # vein
    draw.line([cx - 85, cy, cx + 85, cy], fill=(min(green[0] + 30, 255), min(green[1] + 30, 255), green[2]), width=3)
    return img


def add_blight_spots(img: Image.Image, rng) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(8, 18)):
        x = int(rng.integers(40, IMG_SIZE - 40))
        y = int(rng.integers(60, IMG_SIZE - 60))
        r = int(rng.integers(4, 12))
        brown = (int(rng.integers(90, 130)), int(rng.integers(60, 90)), int(rng.integers(20, 45)))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=brown)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def add_rust_spots(img: Image.Image, rng) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(15, 30)):
        x = int(rng.integers(40, IMG_SIZE - 40))
        y = int(rng.integers(60, IMG_SIZE - 60))
        r = int(rng.integers(2, 6))
        orange = (int(rng.integers(170, 220)), int(rng.integers(90, 130)), int(rng.integers(10, 40)))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=orange)
    return img.filter(ImageFilter.GaussianBlur(0.4))


def make_image(cls: str, rng) -> Image.Image:
    img = make_leaf_base(rng)
    if cls == "leaf_blight":
        img = add_blight_spots(img, rng)
    elif cls == "leaf_rust":
        img = add_rust_spots(img, rng)
    # mild random noise so the classification task isn't trivially easy
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-8, 8, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_split(out_dir: str, split: str, n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    for cls in CLASSES:
        cls_dir = os.path.join(out_dir, split, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = make_image(cls, rng)
            img.save(os.path.join(cls_dir, f"{cls}_{i:04d}.jpg"), quality=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="../data")
    parser.add_argument("--n_per_class", type=int, default=120)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    args = parser.parse_args()

    n_train = int(args.n_per_class * (1 - args.val_fraction))
    n_val = args.n_per_class - n_train

    build_split(args.out, "train", n_train, seed=42)
    build_split(args.out, "val", n_val, seed=99)

    print(f"Generated synthetic dataset: {n_train} train / {n_val} val images per class "
          f"across {len(CLASSES)} classes in {args.out}")
