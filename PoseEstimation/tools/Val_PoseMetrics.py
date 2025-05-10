# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Jake O'Brien
# ------------------------------------------------------------------------------

"""
Val_PoseMetrics.py

# Overview:
#     1. Parse command-line arguments and load the experiment configuration.
#     2. Initialize the logger, models, and validation dataloader.
#     3. Build a ground-truth lookup from the COCO/PERSON JSON annotations.
#     4. For each validation image:
#        a. Run the pose network to predict 2D keypoints.
#        b. Look up the corresponding ground-truth 6DoF pose.
#        c. Load and transform the 3D keypoints from the .mat file.
#        d. Recover the estimated 6DoF pose via EPnP+RANSAC.
#        e. Compute per-image metrics:
#           • Rotation error (degrees)
#           • Translation error (same units as GT)
#           • Combined pose error (E_R + E_T/‖t_gt‖)
#           • MPJPE (2D keypoint reprojection error)
#        f. Accumulate these metrics.
#     5. After all images:
#        a. Print total skips and drop rates.
#        b. Compute average and median of each metric.
#        c. Save the aggregate metrics to JSON.
#
# Usage:
#     python Val_PoseMetrics.py --cfg config.yaml [opts]
#
# Outputs:
#     • Console log of per-image errors and overall summary statistics
#     • `average_pose_errors.json` in the output directory, containing:
#         – avg_rotation_error_deg
#         – avg_translation_error
#         – avg_combined_pose_error
#         – avg_mpjpe
#         – median_rotation_error_deg
#         – median_translation_error
#         – median_combined_pose_error
#         – median_mpjpe  
"""

from __future__ import absolute_import, division, print_function

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import argparse
import os
import pprint
import json
import math

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
from scipy.io import loadmat
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Local application imports
# ------------------------------------------------------------------------------
import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs, aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger, get_model_summary
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale, get_final_preds, get_multi_scale_size

# ------------------------------------------------------------------------------
# Multiprocessing configuration
# ------------------------------------------------------------------------------
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------------------------------------------------------------
# Configuration / Argument Utilities
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Inference and Pose Evaluation on Validation Dataset')
    parser.add_argument('--cfg', help='Experiment configuration file', required=True, type=str)
    parser.add_argument('opts', help="Modify config options via command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info('| Arch ' + ' '.join(['| {}'.format(name) for name in names]) + ' |')
    logger.info('|---' * (num_values+1) + '|')
    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info('| ' + full_arch_name + ' ' + ' '.join(['| {:.3f}'.format(value) for value in values]) + ' |')


def build_gt_dict(val_json_path):
    """
    Builds a dictionary from the validation JSON file, keyed by 'filename'
    instead of 'file_name'. If JSON is in COCO format, 'images' is expected.
    """
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    if isinstance(val_data, dict) and "images" in val_data:
        images = val_data["images"]
    else:
        images = val_data

    gt_dict = {}
    for rec in images:
        fn = rec.get("file_name")
        if fn is None:
            print(f"Warning: skipping record with no filename: {rec}")
            continue
        gt_dict[fn] = rec
    return gt_dict


def parse_pose(raw_data, expected_len):
    """
    Converts either a dict with keys '0', '1', '2', (and '3' if needed)
    or a list of floats into a NumPy array of length expected_len.
    """
    if isinstance(raw_data, dict):
        return np.array([raw_data[str(i)] for i in range(expected_len)], dtype=np.float64)
    else:
        return np.array(raw_data, dtype=np.float64)

# ------------------------------------------------------------------------------
# Pose Error Metrics
# ------------------------------------------------------------------------------
def rotation_error_deg(R_pred, R_gt):
    """
    Compute the angular error (degrees) between two rotation matrices.

    Args:
        R_pred: Predicted rotation matrix (3×3).
        R_gt: Ground-truth rotation matrix (3×3).

    Returns:
        Rotation error in degrees.
    """
    R_err = R_pred @ R_gt.T
    val = (np.trace(R_err) - 1.0) / 2.0
    val_clamped = np.clip(val, -1.0, 1.0)
    theta_rad = np.arccos(val_clamped)
    return np.degrees(theta_rad)

def translation_error(t_pred, t_gt):
    """
    Compute the Euclidean distance between predicted and ground-truth translations.

    Args:
        t_pred: Predicted translation vector (3,).
        t_gt: Ground-truth translation vector (3,).

    Returns:
        Translation error (same units as input).
    """
    return np.linalg.norm(t_pred - t_gt)

def total_pose_error(R_pred, R_gt, t_pred, t_gt):
    """
    Compute combined pose error: rotation + normalized translation.

    Args:
        R_pred: Predicted rotation matrix (3×3).
        R_gt: Ground-truth rotation matrix (3×3).
        t_pred: Predicted translation vector (3,).
        t_gt: Ground-truth translation vector (3,).

    Returns:
        E_pose: E_R + (E_T / ||t_gt||)
        E_R: Rotation error (degrees).
        E_T: Translation error.
    """
    E_R = rotation_error_deg(R_pred, R_gt)
    E_T = translation_error(t_pred, t_gt)
    norm_t_gt = np.linalg.norm(t_gt)
    if norm_t_gt < 1e-8:
        norm_t_gt = 1.0
    E_pose = E_R + (E_T / norm_t_gt)
    return E_pose, E_R, E_T

def compute_mpjpe(pred, gt):
    """
    Compute Mean Per-Joint Position Error (MPJPE) for 2D keypoints.

    Args:
        pred: Predicted keypoints array (N×2).
        gt: Ground-truth keypoints array (N×2).

    Returns:
        Mean Euclidean error across all keypoints.
    """
    errors = np.linalg.norm(pred - gt, axis=1)
    return np.mean(errors)


# ------------------------------------------------------------------------------
# Camera & Geometry Utilities
# ------------------------------------------------------------------------------
class Camera:
    """
    Load and store camera calibration parameters from a JSON file.

    Args:
        root_dir: Directory path containing 'camera.json'.

    Attributes:
        fx, fy: Focal lengths in pixels.
        nu, nv: Sensor resolution (width x height).
        ppx, ppy: Principal point coordinates.
        fpx, fpy: Normalized focal lengths.
        K: 3x3 camera intrinsic matrix.
        dcoef: Distortion coefficients.
    """
    def __init__(self, root_dir):
        with open(os.path.join(root_dir, 'camera.json'), 'r') as f:
            camera_params = json.load(f)
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.nu = camera_params['Nu']
        self.nv = camera_params['Nv']
        self.ppx = camera_params['ppx']
        self.ppy = camera_params['ppy']
        self.fpx = self.fx / self.ppx
        self.fpy = self.fy / self.ppy
        self.k = camera_params['cameraMatrix']
        self.K = np.array(self.k, dtype=np.float32)
        self.dcoef = camera_params['distCoeffs']
        
# ----------------------------
# Quaternion to Rotation Matrix
# ----------------------------
def quat_to_dcm(q):
    """
    Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix.

    Args:
        q: Quaternion as array-like of length 4.

    Returns:
        3x3 rotation matrix.
    """
    q = q / np.linalg.norm(q)
    x, y, z, w = q  # assumes qRotation is [x, y, z, w]
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
    return R

# ----------------------------
# Custom Projection Function
# ----------------------------
def project_keypoints(kpts, R, t, fx, cx, cy):
    """
    Project 3D keypoints into 2D image coordinates using a pinhole camera model.

    Args:
        kpts: Array of 3D points (N x 3).
        R: Rotation matrix (3 x 3).
        t: Translation vector (3,) or (3,1).
        fx: Focal length (pixels).
        cx: Principal point x-coordinate (pixels).
        cy: Principal point y-coordinate (pixels).

    Returns:
        Tuple of u and v coordinate arrays, each of length N.
    """
    cam_pts = (R @ kpts.T).T + t.reshape(1, 3)
    u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
    v = -fx * cam_pts[:, 1] / cam_pts[:, 2] + cy
    return u, v

# ------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------
def main():

    total_keypoints = 0
    total_dropped = 0
    skip_count_visibility = 0

    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    dump_input = torch.rand((1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))
    if cfg.FP16.ENABLED:
        model = network_to_half(model)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, 'model_best.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    # Use the full test dataloader.
    data_loader, test_dataset = make_test_dataloader(cfg)

    # Build a dictionary from the validation JSON (using 'filename').
    val_json_path = "/home/realtra/Documents/HigherHRNet_pose/KRN_SimDataset_COCO/annotations/person_keypoints_val_coco.json"
    gt_dict = build_gt_dict(val_json_path)
    logger.info("Built GT dictionary with {} entries.".format(len(gt_dict)))

    # Define image transformation.
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    parser = HeatmapParser(cfg)

    # Lists to accumulate errors.
    all_rot_err = []
    all_trans_err = []
    all_pose_err = []
    all_mpjpe = []

    # Counter for skipped images.
    skip_count = 0

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None

    # Initialize camera.
    cam = Camera("/home/realtra/Documents/HigherHRNet_pose/KRN_SimDataset_COCO")

    for i, (images, annos) in enumerate(data_loader):
        # Extract filename from the COCO annotations or fallback.
        if hasattr(test_dataset, "coco"):
            img_info = test_dataset.coco.loadImgs(test_dataset.ids[i])[0]
            filename = img_info["file_name"]
            print(f"Processing image: {filename}")
        else:
            filename = f"image_{i}.jpg"
            print(f"Processing image: {filename} (no coco attribute found)")

        image = images[0].cpu().numpy()
        base_size, center, scale = get_multi_scale_size(image, cfg.DATASET.INPUT_SIZE, 1.0,
                                                        min(cfg.TEST.SCALE_FACTOR))

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()
                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )
                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )
            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)

            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )
            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
            
            # Print the final results to inspect the prediction
            #print(f"Final results for {filename}: {final_results}")

            if not final_results:  # if final_results is empty
                print(f"No valid poses found for {filename}; skipping.")
                skip_count += 1
                continue
            
            # Use only the first 11 keypoints (for pose estimation) and the first two columns (x, y).
            predicted_2d_full = final_results[0]
            predicted_2d = predicted_2d_full[:11, :2]
            #print(f"{filename} | Predicted 2D keypoints shape: {predicted_2d.shape}")
            print("Predicted 2D keypoints:\n", predicted_2d)

        # PARAMETERS
        labels_base_dir = "/home/realtra/Documents/KRN_SimDataset"
        subfolders = ["LEO", "MEO", "GEO", "MIX"]
            
        # Remove file extension from filename
        filename_no_ext = os.path.splitext(filename)[0]

        # Search labels.json in all subfolders ---
        record = None
        labels_json_found = None
        for subfolder in subfolders:
            labels_json = os.path.join(labels_base_dir, subfolder, "labels.json")
            with open(labels_json, 'r') as f:
                data = json.load(f)
            for rec in data["images"]:
                if rec["filename"].lower() == filename_no_ext.lower():
                    record = rec
                    labels_json_found = labels_json
                    break
            if record is not None:
                break

        if record is None:
            print(f"Could not find record with filename {filename} in any subfolder.")
            return
        else:
            print(f"Found record in {labels_json_found}")


        # Extract GT Pose from json
        gt_translation = np.array(record["localPosition"], dtype=np.float64)
        gt_quat = np.array(record["qRotation"], dtype=np.float64)
        R_gt = quat_to_dcm(gt_quat)
        t_gt = gt_translation
        print("GT rotation matrix:\n", R_gt)
        print("GT translation vector:\n", t_gt)

        # Load GT 3D Keypoints from .mat
        keypoints_3d_path = "/home/realtra/Documents/SimTANGO_kpts.mat"
        mat = loadmat(keypoints_3d_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        gt_3d_raw = mat[keys[0]]
        if gt_3d_raw.shape[0] == 3:
            gt_3d_raw = gt_3d_raw.T
        gt_3d_raw = gt_3d_raw[:11]  # use only the first 11 keypoints
        print("Raw GT 3D keypoints shape:", gt_3d_raw.shape)
    
        # Load Camera Parameters
        camera_matrix = cam.K
        dist_coeffs = np.array(cam.dcoef, dtype=np.float32) if cam.dcoef is not None else np.zeros((4,1), dtype=np.float32)
        #print("Camera matrix:\n", camera_matrix)
        #print("Distortion coefficients:\n", dist_coeffs)
    
        # Project GT 3D Keypoints using GT Pose with Custom Function
        # Extract fx, cx, cy from camera matrix
        fx = camera_matrix[0, 0]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        u, v = project_keypoints(gt_3d_raw, R_gt, t_gt, fx, cx, cy)
        projected_gt = np.stack([u, v], axis=-1)
        #print("Projected GT 2D keypoints (custom):\n", projected_gt)
        
        # Skip if not all 11 GT keypoints are inside the image
        if not np.all((projected_gt[:,0] >= 0) & (projected_gt[:,0] < cam.nu) &
                      (projected_gt[:,1] >= 0) & (projected_gt[:,1] < cam.nv)):
            print(f"Skipping {filename}: only {np.sum((projected_gt[:,0] >= 0) & (projected_gt[:,0] < cam.nu) & (projected_gt[:,1] >= 0) & (projected_gt[:,1] < cam.nv))}/11 GT keypoints visible")
            skip_count_visibility += 1
            continue

        # Load 3D keypoints
        mat = loadmat(keypoints_3d_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        keypoints_3d = mat[keys[0]]

        if keypoints_3d.shape[0] == 3:
            keypoints_3d = keypoints_3d.T
        keypoints_3d = keypoints_3d[:11]
        
        # Apply winning permutation and sign flip to 3D keypoints
        perm = (1, 2, 0)
        sign = (-1, 1, 1)
        keypoints_3d = keypoints_3d[:, list(perm)] * np.array(sign).reshape(1, 3)
        #print("Transformed GT 3D keypoints:\n", keypoints_3d)

        keypoints_3d[:,1] *= -1 # flip Y
        
        print("Transformed GT 3D keypoints:\n", keypoints_3d)

        N_pts = min(predicted_2d.shape[0], keypoints_3d.shape[0])
        pred_2d_subset = predicted_2d[:N_pts]
        keypoints_3d_subset = keypoints_3d[:N_pts]

        # Solve EPnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            keypoints_3d_subset.astype(np.float32),
            pred_2d_subset.astype(np.float32),
            cam.K,
            np.array(cam.dcoef, dtype=np.float32) if cam.dcoef is not None else np.zeros((4, 1), dtype=np.float32),
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            print(f"EPnP failed for {filename}; skipping.")
            skip_count += 1
            continue

        R_pred, _ = cv2.Rodrigues(rvec)
        t_pred = tvec.flatten()
        
        # Grab only the inlier indices:
        inl = inliers.flatten()
        
        # How many keypoints tried vs how many RANSAC kept
        dropped = N_pts - len(inl)
        total_keypoints += N_pts
        total_dropped += dropped

        print(f"Dropped {dropped}/{N_pts} keypoints for {filename}")


        # Reproject those 3D inliers into 2D using final R_pred and t_pred
        pts3d_inl   = keypoints_3d_subset[inl]
        rvec_float, _ = cv2.Rodrigues(R_pred.astype(np.float32))
        tvec_float    = t_pred.reshape(3,1).astype(np.float32)
        
        # Undo Y‐flip (OpenCV frame to Unity frame)
        E = np.diag([1, -1, 1])
        kpts3d_int = (E @ pts3d_inl.T).T
        R_int  = E @ R_pred @ E
        t_int  = E @ t_pred

        # Undo winning‐config permutation + sign:
        # originkeypoints_3d_subsetal winning config was perm=(1,2,0), sign=(-1,1,1)
        inv_sign = np.array([-1, 1, 1])
        inv_perm = (2, 0, 1)
        # Bring 3D points back to raw ordering
        kpts3d_raw = (kpts3d_int * inv_sign[None, :])[:, inv_perm]

    # Reproject all 11 keypoints in the raw frame
        u_inl, v_inl = project_keypoints(
            kpts3d_raw, R_int, t_int,
            fx=camera_matrix[0,0],
            cx=camera_matrix[0,2],
            cy=camera_matrix[1,2]
    )
        projected_pred_inl = np.stack([u_inl, v_inl], axis=1) # (11,2)

        print("\nCleaned Recovered Rotation Matrix:")
        print(np.round(R_int, 4))
        print("\nCleaned Recovered Translation Vector:")
        print(np.round(t_int, 4))
        
        # DEBUG: Print translation values to check unit consistency.
        #print("GT translation:", t_gt)
        #print("Predicted translation:", t_int)
        
        # DEBUG: print pred vs gt and their difference
        print("projected_pred (11×2):\n", projected_pred_inl)
        print("projected_gt   (11×2):\n", projected_gt)
        #print("difference pred−gt:\n", projected_pred - projected_gt)

        # Compute per-image error metrics.
        rot_err = rotation_error_deg(R_int, R_gt)
        trans_err = translation_error(t_int, t_gt)
        E_pose, E_R, E_T = total_pose_error(R_int, R_gt, t_int, t_gt)
        
        # Compute 2D Keypoint Error (MPJPE) on inliers only
        gt_inl   = projected_gt[inl]
        mpjpe    = compute_mpjpe(projected_pred_inl, gt_inl)

        print(f"{filename} - Rot error: {rot_err:.4f} deg, "
              f"Trans error: {trans_err:.4f}, "
              f"Pose error: {E_pose:.4f}, "
              f"MPJPE: {mpjpe:.4f} pixels")

        all_rot_err.append(rot_err)
        all_trans_err.append(trans_err)
        all_pose_err.append(E_pose)
        all_mpjpe.append(mpjpe)

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    print("Total images skipped:", skip_count)
    print("Total images skipped (GT not fully visible):", skip_count_visibility)

    # Compute average errors over the dataset.
    avg_rot_err = np.mean(all_rot_err) if all_rot_err else float('nan')
    avg_trans_err = np.mean(all_trans_err) if all_trans_err else float('nan')
    avg_pose_err = np.mean(all_pose_err) if all_pose_err else float('nan')
    avg_mpjpe = np.mean(all_mpjpe) if all_mpjpe else float('nan')

    print("Average Rotation error (deg): {:.4f}".format(avg_rot_err))
    print("Average Translation error (units): {:.4f}".format(avg_trans_err))
    print("Average Combined pose error: {:.4f}".format(avg_pose_err))
    print("Average MPJPE (2D keypoint error): {:.4f} pixels".format(avg_mpjpe))
    
    # Compute median errors over the dataset.
    median_rot_err = np.median(all_rot_err) if all_rot_err else float('nan')
    median_trans_err = np.median(all_trans_err) if all_trans_err else float('nan')
    median_pose_err = np.median(all_pose_err) if all_pose_err else float('nan')
    median_mpjpe = np.median(all_mpjpe) if all_mpjpe else float('nan')

    print("Median Rotation error (deg): {:.4f}".format(median_rot_err))
    print("Median Translation error (units): {:.4f}".format(median_trans_err))
    print("Median Combined pose error: {:.4f}".format(median_pose_err))
    print("Median MPJPE (2D keypoint error): {:.4f} pixels".format(median_mpjpe))

    avg_error_results = {
        "avg_rotation_error_deg": avg_rot_err,
        "avg_translation_error": avg_trans_err,
        "avg_combined_pose_error": avg_pose_err,
        "avg_mpjpe": float(avg_mpjpe),
        "median_rotation_error_deg": median_rot_err,
        "median_translation_error": median_trans_err,
        "median_combined_pose_error": median_pose_err,
        "median_mpjpe": float(median_mpjpe)
    }
    
    print(f"\n=== Keypoint drop summary ===")
    print(f"Total keypoints processed: {total_keypoints}")
    print(f"Total keypoints dropped : {total_dropped}")
    print(f"Overall drop rate        : {100 * total_dropped/total_keypoints:.2f}%")
    
    avg_error_save_path = os.path.join(final_output_dir, "average_pose_errors.json")
    with open(avg_error_save_path, 'w') as f:
        json.dump(avg_error_results, f, indent=4)
    print("Saved average and median pose error metrics to:", avg_error_save_path)


if __name__ == "__main__":
    main()
