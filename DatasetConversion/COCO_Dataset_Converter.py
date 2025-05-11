# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O’Brien (extended)
# ------------------------------------------------------------------------------

"""
COCO_Dataset_Converter.py

Overview:
    Converts a custom simulated dataset into COCO-format keypoint annotations
    and train/val image splits.

Configuration:
    - Set `dataset_root` to the root of your source images.
    - Set `target_root` to the output directory for COCO images and annotations.
    - Set `mat_file_path` to the .mat file containing 3D keypoints.
    - Define `folders` (['GEO','LEO','MEO','MIX']) under `dataset_root`.

Outputs:
    • COCO JSON annotation files:
        - person_keypoints_train_coco.json
        - person_keypoints_val_coco.json
    • Image folders:
        - images/train_coco/
        - images/val_coco/
    • camera.json under `target_root` containing intrinsics.
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import os
import json
import shutil
import random
from math import tan, radians

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import numpy as np
from scipy.io import loadmat

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
random.seed(2010)
dataset_root = '/home/realtra/Documents/KRN_SimDataset'
target_root = '/home/realtra/Documents/HigherHRNet_pose/KRN_SimDataset_COCO'
mat_file_path = '/home/realtra/Documents/SimTANGO_kpts.mat'
folders = ['GEO', 'LEO', 'MEO', 'MIX']
default_fov = 60  # degrees

# ------------------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------------------
annotations_dir = os.path.join(target_root, 'annotations')
train_img_dir = os.path.join(target_root, 'images', 'train_coco')
val_img_dir = os.path.join(target_root, 'images', 'val_coco')
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load and preprocess 3D keypoints
# ------------------------------------------------------------------------------
mat = loadmat(mat_file_path)
key = [k for k in mat.keys() if not k.startswith("__")][0]
keypoints_3d = mat[key]  # (3, N) or (N, 3)

# Transpose to (N, 3) if necessary
if keypoints_3d.shape[0] == 3:
    keypoints_3d = keypoints_3d.T

# Apply winning permutation and sign flip
perm = (1, 2, 0)
sign = (-1, 1, 1)
keypoints_3d = keypoints_3d[:, list(perm)] * np.array(sign).reshape(1, 3)

num_keypoints = keypoints_3d.shape[0]

# COCO category template
categories = [{
    "id": 1,
    "name": "TANGO",
    "keypoints": [f"kp{i}" for i in range(17)],  # Expecting 17 keypoints in total
    "skeleton": []
}]

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def quat_to_rotmat(q):
    """
    Convert a quaternion [x, y, z, w] to a 3×3 rotation matrix.

    Args:
        q: Iterable of length 4 (x, y, z, w).

    Returns:
        R: NumPy array shape (3, 3).
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
    ])

def project_keypoints(kpts, R, t, fx, cx, cy):
    """
    Project 3D keypoints into 2D image coordinates via a pinhole camera model.

    Args:
        kpts: (N, 3) array of 3D points.
        R:    (3, 3) rotation matrix.
        t:    (3,) translation vector.
        fx:   focal length in pixels.
        cx:   principal point x-coordinate.
        cy:   principal point y-coordinate.

    Returns:
        u, v: 1D arrays of length N with image coordinates.
    """
    cam_pts = (R @ kpts.T).T + t.reshape(1, 3)
    u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
    v = -fx * cam_pts[:, 1] / cam_pts[:, 2] + cy
    return u, v

# === PAD KEYPOINTS TO MATCH 17 ===
def pad_keypoints(keypoints):
    """
    Pad a flat list of keypoints [x1,y1,v1, x2,y2,v2, ...] to 17 points.

    Missing keypoints are filled as [0,0,0] (invisible).

    Args:
        keypoints: List of floats, length 3*K where K < 17.

    Returns:
        Padded list of length 3*17.
    """
    # Add empty keypoints (set to [0, 0, 0]) if there are fewer than 17
    while len(keypoints) < 51:  # 51 entries (3 per keypoint)
        keypoints.extend([0, 0, 0])  # Adding empty keypoints (x=0, y=0, visibility=0)
    return keypoints

# ------------------------------------------------------------------------------
# Prepare COCO structures
# ------------------------------------------------------------------------------
img_id, ann_id = 0, 0
train_images, val_images = [], []
train_annotations, val_annotations = [], []

# ------------------------------------------------------------------------------
# Process each folder, copy images, build annotations
# ------------------------------------------------------------------------------
for folder in folders:
    folder_path = os.path.join(dataset_root, folder)
    labels_path = os.path.join(folder_path, "labels.json")

    with open(labels_path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    img_w, img_h = config.get("imageWidth", 640), config.get("imageHeight", 640)
    fov = config.get("fov", 60)
    fx = (img_w / 2) / tan(radians(fov / 2))
 
    images = data["images"]
    random.shuffle(images)
    split_idx = int(0.8 * len(images))
    split_data = [("train", images[:split_idx]), ("val", images[split_idx:])]

    for split_name, entries in split_data:
        for entry in entries:
            fname = entry["filename"] + ".png"
            src_path = os.path.join(folder_path, fname)
            dst_dir = train_img_dir if split_name == "train" else val_img_dir
            dst_path = os.path.join(dst_dir, fname)

            if not os.path.exists(src_path):
                print(f"❌ Missing image: {src_path}")
                continue
            shutil.copy(src_path, dst_path)

            # Pose and projection
            R = quat_to_rotmat(entry["qRotation"])
            t = np.array(entry["localPosition"])
            u, v = project_keypoints(keypoints_3d, R, t, fx, img_w / 2, img_h / 2)

            # Build keypoints list and pad
            keypoints = []
            for x, y in zip(u, v):
                keypoints.extend([float(x), float(y), 2])

            # Pad keypoints to match 17 keypoints
            keypoints = pad_keypoints(keypoints)

            # BBox and area
            x0, y0, x1, y1 = entry["boundingBox"]
            bbox = [x0, y0, x1 - x0, y1 - y0]
            area = bbox[2] * bbox[3]

            # Image info
            img_info = {
                "id": img_id,
                "file_name": fname,
                "width": img_w,
                "height": img_h
            }
            ann_info = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 17  # Set the num_keypoints to 17
            }

            if split_name == "train":
                train_images.append(img_info)
                train_annotations.append(ann_info)
            else:
                val_images.append(img_info)
                val_annotations.append(ann_info)

            img_id += 1
            ann_id += 1

# === SAVE ANNOTATIONS ===
with open(os.path.join(annotations_dir, "person_keypoints_train_coco.json"), "w") as f:
    json.dump({"images": train_images, "annotations": train_annotations, "categories": categories}, f, indent=4)

with open(os.path.join(annotations_dir, "person_keypoints_val_coco.json"), "w") as f:
    json.dump({"images": val_images, "annotations": val_annotations, "categories": categories}, f, indent=4)

print("COCO annotations saved.")

# ------------------------------------------------------------------------------
# Save camera intrinsics
# ------------------------------------------------------------------------------
camera_info = {
    "Nu": img_w,
    "Nv": img_h,
    "ppx": 5.86e-6,
    "ppy": 5.86e-6,
    "fx": fx * 5.86e-6,
    "fy": fx * 5.86e-6,
    "ccx": img_w / 2,
    "ccy": img_h / 2,
    "cameraMatrix": [
        [fx, 0, img_w / 2],
        [0, fx, img_h / 2],
        [0, 0, 1]
    ],
    "distCoeffs": [0, 0, 0, 0, 0]
}
with open(os.path.join(target_root, "camera.json"), "w") as f:
    json.dump(camera_info, f, indent=4)

print("camera.json saved.")