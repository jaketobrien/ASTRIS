# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O’Brien (extended)
# ------------------------------------------------------------------------------

"""
YOLO_Dataset_Converter.py

Overview:
    Restructures simulated dataset for YOLO object detection.
    - Splits images into train/val sets.
    - Converts bounding-box annotations to YOLO format.
    - Organizes images and labels into appropriate folders.

Configuration:
    - `source_root`: Path to the root of your source dataset.
    - `output_root`: Path where the YOLO-formatted dataset will be created.
    - `split_ratio`: Fraction of images to use for training (e.g., 0.8).
    - `img_ext`: Image file extension (e.g., ".png").

Outputs:
    • Folder structure under `output_root`:
        - train/images/, train/labels/
        - val/images/,   val/labels/
    • For each image, a corresponding `.txt` file with YOLO-format bbox.
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import os
import json
import shutil
from pathlib import Path

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
source_root = Path("/home/realtra/Documents/SDN_SimDataset")  # <-- adjust if needed
output_root = Path("/home/realtra/Documents/YOLO11_OD/SimDataset_YOLO")
split_ratio = 0.8
img_ext = ".png"

# ------------------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------------------
train_images_dir = output_root / "train" / "images"
train_labels_dir = output_root / "train" / "labels"
val_images_dir = output_root / "val" / "images"
val_labels_dir = output_root / "val" / "labels"

for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def convert_bbox(bbox, img_width, img_height):
    """
    Convert a bounding box to YOLO format: class x_center y_center width height.

    Args:
        bbox: Tuple (x_min, y_min, x_max, y_max) in pixel coordinates.
        img_width:  Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        A string "0 x_center y_center width height" with normalized floats.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# ------------------------------------------------------------------------------
# Main processing loop
# ------------------------------------------------------------------------------

# === Process each batch folder ===
image_id = 0
for folder in ["LEO", "MEO", "GEO", "MIX"]:
    folder_path = source_root / folder
    labels_file = folder_path / "labels.json"
    if not labels_file.exists():
        print(f"⚠️ Missing labels.json in {folder_path}")
        continue

    with open(labels_file, "r") as f:
        data = json.load(f)

    img_width = data["config"]["imageWidth"]
    img_height = data["config"]["imageHeight"]
    all_items = data["images"]

    # Split into train and validation subsets
    train_items, val_items = train_test_split(all_items, train_size=split_ratio, random_state=42)

    for subset, items, img_out_dir, lbl_out_dir in [
        ("train", train_items, train_images_dir, train_labels_dir),
        ("val", val_items, val_images_dir, val_labels_dir)
    ]:
        for item in items:
            old_name = item["filename"] + img_ext
            new_name = f"img{image_id:06d}"
            src_img = folder_path / old_name
            dst_img = img_out_dir / (new_name + img_ext)
            dst_lbl = lbl_out_dir / (new_name + ".txt")

            if not src_img.exists():
                print(f"❌ Image not found: {src_img}")
                continue
            
            # Copy image
            shutil.copy(src_img, dst_img)

            # Convert and save YOLO-format bbox
            bbox_yolo = convert_bbox(item["boundingBox"], img_width, img_height)
            with open(dst_lbl, "w") as out_lbl:
                out_lbl.write(bbox_yolo + "\n")

            image_id += 1

print("Dataset conversion complete.")