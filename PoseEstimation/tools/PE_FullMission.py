# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Jake O'Brien
# ------------------------------------------------------------------------------

"""
PE_FullMission.py

Overview:
    1. Parse command-line arguments and load the experiment configuration.
    2. Initialize logging and directories (raw frames, detections, crops,
       keypoints, CSVs).
    3. Load YOLO detector and pose-estimation network (HigherHRNet) onto GPU.
    4. Open a TCP socket and wait for incoming simulation frames and
       ground-truth poses.
    5. For each frame:
       a. Receive and decode the image and GT pose (translation + quaternion)
          over TCP.
       b. If in the Near Range phase, run YOLO to detect the object, save a
          bounding-box image, and crop to 640×640.
       c. Project raw 3D keypoints into 2D for GT visualization.
       d. Run the pose model to predict 2D keypoints and save them to .npy.
       e. Use EPnP+RANSAC to recover the 6DoF pose; skip the frame on failure.
       f. Estimate measurement noise via reprojection residuals.
       g. Filter translation and rotation estimates with two Kalman filters.
       h. Compute error metrics: rotation error, translation error, combined
          pose error, and MPJPE.
       i. Log metrics to CSVs and push a visualization payload to the display
          process.
    6. On shutdown (Ctrl+C), cleanly close files, sockets, and print an
       aggregate drop/error summary.

Key components:
    • **Networking** (`socket` + `recvall`): streams sim frames and GT pose over
      TCP  
    • **Detection** (YOLO): crops the image to focus on the target object  
    • **Pose inference** (HigherHRNet & EPnP+RANSAC): predicts 2D keypoints and
      recovers 3D pose  
    • **Filtering** (Kalman): smooths translation and orientation estimates  
    • **Logging** (`Logger`): writes per-frame CSVs (orientation, position,
      keypoints, Kalman state)  
    • **Visualization** (`start_visualizer`): real-time display of frames and
      errors  

Usage:
    python PE_FullMission.py --cfg config.yaml [opts]

Outputs:
    • Per-frame rotation, translation, and pose errors  
    • MPJPE (2D keypoint error)  
    • CSV logs under `results_log/`  
    • Raw input frames under `output_frames/raw/`  
    • Bounding-box detection images under `bbox/`  
    • Cropped images under `crop/`  
    • Predicted keypoints as .npy files under `output_frames/keypoints/`  
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
import socket
import struct
import csv

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import cv2
import scipy.stats
from scipy.io import loadmat
import imageio
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing

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
from dataset import make_live_stream_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger, get_model_summary
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale, get_final_preds, get_multi_scale_size
from multiprocessing import Queue
from vis import start_visualizer
from log import Logger

# ------------------------------------------------------------------------------
# Multiprocessing configuration
# ------------------------------------------------------------------------------
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------------------------------------------------------------
# Kalman Filter Utilities
# ------------------------------------------------------------------------------
def make_t_cv_kf(dt, process_var, meas_var):
    """
    Create a constant-velocity Kalman filter for 3D translation tracking.

    The state vector is 6-dimensional: [x, y, z, vx, vy, vz].

    Args:
        dt: Time step (seconds) between filter updates.
        process_var: Process variance (q) for the motion model.
        meas_var: Measurement variance (r) for observations.

    Returns:
        Configured KalmanFilter instance with:
        - F: state-transition matrix
        - H: measurement matrix (observing position only)
        - Q: process covariance
        - R: measurement covariance
        - P: initial state covariance
        - x: initial state vector (zeros)
    """
    # 6-state: [ x, y, z, vx, vy, vz ]
    kf = KalmanFilter(dim_x=6, dim_z=3)
    
    # State‐transition F
    kf.F = np.eye(6)
    kf.F[0,3] = dt
    kf.F[1,4] = dt
    kf.F[2,5] = dt
    
    # Measurement matrix H: we only observe positions
    kf.H = np.zeros((3,6))
    kf.H[0,0] = 1
    kf.H[1,1] = 1
    kf.H[2,2] = 1
    
    # Covariances
    kf.P *= 1.0 # initial state covariance
    q = process_var
    dt2 = dt*dt
    dt3 = dt2*dt
    dt4 = dt3*dt
    
    # [[dt⁴/4,  dt³/2], [dt³/2, dt²]] for each axis
    Q_block = q * np.block([
      [np.eye(3)*(dt4/4.0), np.eye(3)*(dt3/2.0)],
      [np.eye(3)*(dt3/2.0), np.eye(3)*dt2       ]
    ])
    kf.Q = Q_block

    # measurement noise: tune this to your EPnP residuals
    kf.R = np.eye(3) * meas_var

    # initial state at rest
    kf.x = np.zeros((6,1))
    return kf
    
def make_rot_cv_kf(dt, process_var, meas_var):
    """
    Create a constant-velocity Kalman filter for 3D rotational tracking.

    The state vector is 6-dimensional: [rx, ry, rz, wx, wy, wz],
    where r is the Rodrigues rotation vector and w its angular velocity.

    Args:
        dt: Time step (seconds) between filter updates.
        process_var: Process variance for the rotation model.
        meas_var: Measurement variance for orientation observations.

    Returns:
        Configured KalmanFilter instance with state-transition, measurement,
        process and measurement covariances, and zero initial state.
    """
    # 6‐state: [rx, ry, rz, ωx, ωy, ωz] where r is Rodrigues‐vector, ω its rate
    kf = KalmanFilter(dim_x=6, dim_z=3)
    
    # state‐transition
    kf.F = np.eye(6)
    kf.F[0,3] = dt
    kf.F[1,4] = dt
    kf.F[2,5] = dt
    
    # we only measure the rotation vector r    
    kf.H = np.zeros((3,6))
    kf.H[0,0] = 1
    kf.H[1,1] = 1
    kf.H[2,2] = 1
    
    # covariances    
    kf.P *= 1.0
    dt2, dt3, dt4 = dt*dt, dt**3, dt**4
    Q_block = process_var * np.block([
        [np.eye(3)*(dt4/4), np.eye(3)*(dt3/2)],
        [np.eye(3)*(dt3/2), np.eye(3)*dt2]
    ])
    kf.Q = Q_block
    kf.R = np.eye(3) * meas_var
    kf.x = np.zeros((6,1))
    return kf

def update_cv_kf(kf, dt, process_var):
    """
    Update the state-transition and process covariance of a constant-velocity
    Kalman filter given a new time step.

    Args:
        kf: Existing KalmanFilter instance to modify.
        dt: New time step (seconds) for update.
        process_var: Process variance parameter for recalculating Q.
    """
    # Update transition matrix for new dt
    kf.F[0,3] = kf.F[1,4] = kf.F[2,5] = dt
    
    # Recompute process covariance Q
    dt2, dt3, dt4 = dt*dt, dt**3, dt**4
    kf.Q = process_var * np.block([
        [np.eye(3)*(dt4/4), np.eye(3)*(dt3/2)],
        [np.eye(3)*(dt3/2), np.eye(3)*dt2]
    ])
   
# ------------------------------------------------------------------------------
# Networking Utilities
# ------------------------------------------------------------------------------
def recvall(sock, n):
    """
    Receive exactly n bytes from a socket or raise ConnectionError.

    Args:
        sock: A connected socket object.
        n: Number of bytes to receive.

    Returns:
        A bytes object of length n.

    Raises:
        ConnectionError: If the socket closes before n bytes are received.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed during recv")
        data += packet
    return data
    
# ------------------------------------------------------------------------------
# Configuration / Argument Utilities
# ------------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments for experiment configuration.

    Returns:
        An argparse.Namespace containing:
        - cfg: Path to the YAML config file (required).
        - opts: Additional config overrides (list of strings).
        - port: TCP port to listen on (default: 5000).
    """
    parser = argparse.ArgumentParser(description='Inference and Pose Evaluation on Validation Dataset')
    parser.add_argument('--cfg', help='Experiment configuration file', required=True, type=str)
    parser.add_argument('opts', help="Modify config options via command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--port', help='TCP port to listen on', default=5000, type=int)
    args = parser.parse_args()
    return args


def parse_pose(raw_data, expected_len):
    """
    Normalize pose data from a dict or list into a fixed-length array.

    Args:
        raw_data: Either:
            - dict mapping string indices '0','1',… to floats
            - flat list of floats
        expected_len: Number of elements to extract.

    Returns:
        A NumPy array of shape (expected_len,) and dtype float64.
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

def rotation_matrix_to_euler(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw) in degrees.

    Uses the 'xyz' convention (roll-pitch-yaw).

    Args:
        R: Rotation matrix (3×3).

    Returns:
        Array [rx, ry, rz] in degrees.
    """
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2( R[2,1],  R[2,2])
        ry = math.atan2(-R[2,0],  sy)
        rz = math.atan2( R[1,0],  R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0],  sy)
        rz = 0
    return np.degrees([rx, ry, rz])


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

    # ------------------------------------------------------------------------------
    # Constants & Paths
    # ------------------------------------------------------------------------------
    YOLO_MODEL_PATH = "YOLO11s_trained.pt"
    keypoints_3d_path  = "/home/robin/Documents/SimTANGO_kpts.mat"
    camera_json  = "/home/robin/Documents"
    SAVE_DIR     = "outputs_frames"
    LOG_DIR     = "results_log"
    CROP_DIR     = "crops"
    BBOX_DIR = "bbox"
    RAW_DIR       = os.path.join(SAVE_DIR, "raw")
    OVERLAY_DIR   = os.path.join(SAVE_DIR, "overlay")
    KP_DIR        = os.path.join(SAVE_DIR, "keypoints")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    os.makedirs(KP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(BBOX_DIR, exist_ok=True)
    
    # ------------------------------------------------------------------------------
    # CSV Setup
    # ------------------------------------------------------------------------------
    CONF_CSV = "detection_confidences.csv"
    csv_file = open(CONF_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    # write header
    csv_writer.writerow(["frame_idx", "x1", "y1", "x2", "y2", "confidence", "preprocess_ms", "inference_ms", "postprocess_ms"])
    
    # ------------------------------------------------------------------------------
    # Logger & Config
    # ------------------------------------------------------------------------------
    data_logger = Logger(output_dir=LOG_DIR)

    total_keypoints = 0
    total_dropped = 0
    skip_count_visibility = 0

    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'tcp_recv')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # ------------------------------------------------------------------------------
    # Device & Model Setup
    # ------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # YOLO detector for cropping
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.to(device)

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
    all_preds = []
    all_scores = []
    
    # Lists to accumulate errors.
    all_rot_err = []
    all_trans_err = []
    all_pose_err = []
    all_mpjpe = []

    skip_count = 0 # Counter for skipped images
    drop_count = 0   # network / decoding drops

    # Initialize camera.
    cam = Camera(camera_json)
    
    vis_queue = Queue(maxsize=2)
    viz_proc = start_visualizer(vis_queue)
    
    # Load Camera Parameters
    camera_matrix = cam.K
    dist_coeffs   = np.array(cam.dcoef, dtype=np.float32) \
    if cam.dcoef is not None else np.zeros((4,1), dtype=np.float32)
    
    # 3D keypoints sorted once for duration
    mat           = loadmat(keypoints_3d_path)
    keys          = [k for k in mat.keys() if not k.startswith("__")]
    keypoints_3d  = mat[keys[0]]
    if keypoints_3d.shape[0] == 3:
        keypoints_3d = keypoints_3d.T
    keypoints_3d = keypoints_3d[:11]
    raw_3d_gt = keypoints_3d

    perm = (1, 2, 0)
    sign = (-1, 1, 1)
    keypoints_3d = keypoints_3d[:, list(perm)] * np.array(sign).reshape(1, 3)
    keypoints_3d[:,1] *= -1
    
    # TCP Set up
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('', args.port))
    srv.listen(1)
    logger.info(f"Waiting for TCP connection on port {args.port}…")
    conn, addr = srv.accept()
    logger.info(f"Connected by {addr}")

    prev_state = {'t': None, 'q': None, 'time': None}
    
    ## Kalman Variables
    
    # Translation KF
    initial_dt = 1.0 / 1.0   # e.g. 30 Hz camera
    PROCESS_VAR_T = 1e-1        # tune to your dynamics
    MEAS_VAR_T = 7e-4        # tune to your EPnP uncertainty
    kf_t = make_t_cv_kf(initial_dt, PROCESS_VAR_T, MEAS_VAR_T)
    
    # Rotation KF
    initial_dt = 1.0 / 1.0   # e.g. 30 Hz camera
    PROCESS_VAR_R = 1
    MEAS_VAR_R = 7e-4        # tune to your EPnP uncertainty
    kf_r = make_rot_cv_kf(initial_dt, PROCESS_VAR_R, MEAS_VAR_R)

    prev_time = None
    
    frame_idx = 0
    frame_num = 0
    try:
        while True:
        
            try:
                pose_buf = recvall(conn, 8 + 7*4)
            except ConnectionError:
                logger.warning("Network read failed (pose) → dropping frame")
                drop_count += 1
                continue
            sim_time, px, py, pz, qx, qy, qz, qw = struct.unpack('>d7f', pose_buf)   
            
            
            # Read the 4-byte length header
            try:
                hdr = recvall(conn, 4)
            except ConnectionError:
                logger.warning("Network read failed → dropping frame")
                drop_count += 1
                continue
            length = struct.unpack('>I', hdr)[0]
            
            # Read the image bytes
            try:
                data = recvall(conn, length)
            except ConnectionError:
                logger.warning("Network read failed during payload → dropping frame")
                drop_count += 1
                continue
            
            # Decode to BGR
            buf = np.frombuffer(data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            orig_frame = frame_bgr.copy()
            
            if frame_bgr is None:
                logger.warning("Failed to decode PNG → dropping frame")
                drop_count += 1
                continue         
                
            # Crop to 640×640 if needed
            did_crop = False
            crop_offset = (0, 0)
            h, w = frame_bgr.shape[:2]
            if (h, w) == (1024, 1024):
                # run YOLO on full‐res
                results = yolo(frame_bgr) # Run detection
                boxes   = results[0].boxes # Grab the BoxList
                xyxy    = boxes.xyxy.cpu().numpy().astype(int) # (N,4) array of ints
                confs   = boxes.conf.cpu().numpy() 
                speeds      = results[0].speed # Dict with keys 'preprocess','inference','postprocess'
                pre_ms      = speeds["preprocess"]
                inf_ms      = speeds["inference"]
                post_ms     = speeds["postprocess"]

                if len(xyxy) == 0:
                    logger.warning("No detection → skipping frame")
                    continue
                    
                # Pick first detection
                x1, y1, x2, y2 = xyxy[0]
                conf = confs[0] 
                
                csv_writer.writerow([frame_idx, x1, y1, x2, y2, f"{conf:.4f}", f"{pre_ms:.4f}", f"{inf_ms:.4f}", f"{post_ms:.4f}"])
                csv_file.flush() 
                
               # Save annotated full‐res with bbox
                x1, y1, x2, y2 = xyxy[0]
                conf           = confs[0]
                annot = orig_frame.copy()
                cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annot,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
               
                bbox_filename = os.path.join(BBOX_DIR, f"frame{frame_idx:04d}_bbox_conf{conf:.2f}.png")
                cv2.imwrite(bbox_filename, annot)
                print(f"[SAVE BBOX] {bbox_filename}", end="\r")
                
                ## CROP
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                half = 640 // 2

                # Compute window in the original image
                x0 = cx - half
                y0 = cy - half
                x1_ = cx + half
                y1_ = cy + half
                
                # Clamp to image
                x0_clamped = max(0, x0)
                y0_clamped = max(0, y0)
                x1_clamped = min(w, x1_)
                y1_clamped = min(h, y1_)
                
                # Size of the patch we will actually copy
                h_patch = y1_clamped - y0_clamped
                w_patch = x1_clamped - x0_clamped
                
                crop = np.zeros((640,640,3), dtype=frame_bgr.dtype)
                dx = max(0, -x0)
                dy = max(0, -y0)
                crop[dy:dy + h_patch, dx:dx + w_patch] = frame_bgr[y0_clamped:y1_clamped, x0_clamped:x1_clamped]
                frame_bgr = crop  
                crop_offset = (x0_clamped, y0_clamped)
                did_crop = True    
                
                # Save crop
                crop_filename = os.path.join(
                CROP_DIR,
                f"frame{frame_idx:04d}_det0_conf{conf:.2f}.png")
                cv2.imwrite(crop_filename, crop)
                print(f"[SAVE] {crop_filename}", end="\r")

            # Extract GT Pose GT Info
            gt_translation = np.array([px,py,pz], dtype=np.float64)
            gt_quat = np.array([qx, qy, qz, qw], dtype=np.float64)
            R_gt = quat_to_dcm(gt_quat)
            t_gt = gt_translation
            
            print(f"Frame {frame_idx:06d} GT pos=({px:.3f},{py:.3f},{pz:.3f}) rot=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})")
            
            # Extract fx, cx, cy from camera matrix
            fx = camera_matrix[0, 0]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            u, v = project_keypoints(raw_3d_gt, R_gt, t_gt, fx, cx, cy)
                        
            projected_gt = np.stack([u, v], axis=-1)
            #print("Projected GT 2D keypoints (custom):\n", projected_gt)
            
            filename = f"frame_{frame_idx:06d}"

            # Convert & save raw RGB for debugging
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_path = os.path.join(RAW_DIR, f"raw_{filename}.png")
            imageio.imwrite(raw_path, frame_rgb)
            
            base_size, center, scale = get_multi_scale_size(
                frame_rgb, cfg.DATASET.INPUT_SIZE, 1.0,
                min(cfg.TEST.SCALE_FACTOR)
            )

            with torch.no_grad():
                final_heatmaps = None
                tags_list = []
                for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                    input_size = cfg.DATASET.INPUT_SIZE
                    image_resized, center, scale = resize_align_multi_scale(
                        frame_rgb, input_size, s,
                        min(cfg.TEST.SCALE_FACTOR)  
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

                if not final_results:  # if final_results is empty
                    print(f"No valid poses found for {filename}; skipping.")
                    skip_count += 1
                    continue

                # Use only the first 11 keypoints (for pose estimation) and the first two columns (x, y).
                predicted_2d_full = final_results[0]
                predicted_2d      = predicted_2d_full[:11, :2]
                #print("Predicted 2D keypoints:\n", predicted_2d)
                npy_path = os.path.join(KP_DIR, f"{filename}.npy")
                np.save(npy_path, predicted_2d)
                #print(f"Saved 2D keypoints → {npy_path}")

            #print("Transformed GT 3D keypoints:\n", keypoints_3d)

            N_pts = min(predicted_2d.shape[0], keypoints_3d.shape[0])
            pred_2d_subset = predicted_2d[:N_pts]
            keypoints_3d_subset = keypoints_3d[:N_pts]

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                keypoints_3d_subset.astype(np.float32),
                pred_2d_subset.astype(np.float32),
                cam.K,
                np.array(cam.dcoef, dtype=np.float32) \
                    if cam.dcoef is not None else np.zeros((4, 1), dtype=np.float32),
                flags=cv2.SOLVEPNP_EPNP
            )
            if not success:
                print(f"EPnP failed for {filename}; skipping.")
                skip_count += 1
                continue

            R_pred, _ = cv2.Rodrigues(rvec)
            t_pred = tvec.flatten()
            
            ## Kalman
            #####################################################
            
            # After solvePnPRansac:
            if not success: continue
            pts3d = keypoints_3d_subset[inliers.flatten()]
            pts2d = pred_2d_subset[inliers.flatten()]

            # projectPoints for noise estimation
            proj2d, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, tvec,
                               cam.K, dist_coeffs)
            proj2d = proj2d.reshape(-1,2)
            residuals = np.linalg.norm(pts2d - proj2d, axis=1)
            rms_px = np.sqrt((residuals**2).mean())
            print(f"EPnP reprojection RMS = {rms_px:.2f} px")

            # Convert to meters along depth
            depths = ((R_pred @ pts3d.T).T + t_pred).T[2]
            mean_depth = depths.mean()

            # Focal length in px: fx = cam.K[0,0]
            sigma_m = (rms_px / cam.K[0,0]) * mean_depth
            print(f"Converted measurement σ ≈ {sigma_m:.4f} m → variance r = {sigma_m**2:.6f} m²")
            
            # Compute dt only once per frame
            dt = sim_time - (prev_time or sim_time)
            prev_time = sim_time

            # Common measurement variance
            meas_var = sigma_m**2

            # --- TRANSLATION KF UPDATE ---
            kf_t.R = np.eye(3) * meas_var
            update_cv_kf(kf_t, dt, PROCESS_VAR_T)
            kf_t.predict()
            kf_t.update(t_pred.reshape(3,1))
            t_pred_filt = kf_t.x[:3].flatten()

            # --- ROTATION KF UPDATE ---
            kf_r.R = np.eye(3) * meas_var
            update_cv_kf(kf_r, dt, PROCESS_VAR_R)
            kf_r.predict()
            kf_r.update(rvec.reshape(3,1))
            rvec_filt = kf_r.x[:3].flatten()
            R_pred_filt, _ = cv2.Rodrigues(rvec_filt)

            ################################################
                        
            inl = inliers.flatten()
            dropped = N_pts - len(inl)
            total_keypoints += N_pts
            total_dropped += dropped

            print(f"Dropped {dropped}/{N_pts} keypoints for {filename}")

            pts3d_inl = keypoints_3d_subset[inl]
            rvec_float, _ = cv2.Rodrigues(R_pred.astype(np.float32))
            tvec_float = t_pred.reshape(3,1).astype(np.float32)

            E          = np.diag([1, -1, 1])
            kpts3d_int = (E @ pts3d_inl.T).T
            R_int      = E @ R_pred @ E
            t_int      = E @ t_pred

            inl = inliers.flatten()
            dropped = N_pts - len(inl)
            total_keypoints += N_pts
            total_dropped += dropped

            print(f"Dropped {dropped}/{N_pts} keypoints for {filename}")

            pts3d_inl = keypoints_3d_subset[inl]
            rvec_float, _ = cv2.Rodrigues(R_pred.astype(np.float32))
            tvec_float = t_pred.reshape(3,1).astype(np.float32)

            E          = np.diag([1, -1, 1])
            kpts3d_int = (E @ pts3d_inl.T).T
            R_int      = E @ R_pred @ E
            t_int      = E @ t_pred
            t_int_filt = E @ t_pred_filt
            R_int_filt = E @ R_pred_filt @ E

            inv_sign = np.array([-1, 1, 1])
            inv_perm = (2, 0, 1)
            kpts3d_raw = (kpts3d_int * inv_sign[None, :])[:, inv_perm]

            u_inl, v_inl = project_keypoints(
                kpts3d_raw, R_int, t_int,
                fx=camera_matrix[0,0], cx=camera_matrix[0,2], cy=camera_matrix[1,2]
            )
            projected_pred_inl = np.stack([u_inl, v_inl], axis=1)
            

            print("\nCleaned Recovered Rotation Matrix:")
            print(np.round(R_int, 4))
            print("\nCleaned Recovered Translation Vector:")
            print(np.round(t_int, 4))
            print("projected_pred (11×2):\n", projected_pred_inl)           
            
            # now convert R_int into quaternion (x, y, z, w)
            # Get the axis-angle representation via Rodrigues
            rvec, _ = cv2.Rodrigues(R_int_filt.astype(np.float64))
            θ = np.linalg.norm(rvec)
            if θ < 1e-6:
                # No rotation
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                axis = (rvec.flatten() / θ)
                half = θ / 2.0
                q_xyz = axis * np.sin(half)
                q_w   = np.cos(half)
                q     = np.hstack((q_xyz, q_w))

            print("Recovered quaternion [x, y, z, w]:", np.round(q, 6))
            
            trans_errs = np.abs(t_int_filt - t_gt) # array [err_x, err_y, err_z]
            
            R_err = R_int_filt @ R_gt.T
            rot_errs = rotation_matrix_to_euler(R_err) # [err_roll, err_pitch, err_yaw]

            # Compute per-image error metrics.
            rot_err = rotation_error_deg(R_int, R_gt)
            rot_err_filt = rotation_error_deg(R_int_filt, R_gt)
            trans_err = translation_error(t_int, t_gt)
            trans_err_filt = translation_error(t_int_filt, t_gt)
            E_pose, E_R, E_T = total_pose_error(R_int_filt, R_gt, t_int_filt, t_gt)
            
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
            
            
            # build the dict exactly as vis.py expects:
            payload = {
                # 1) Core pose info
                'frame':            frame_bgr,
                'pred_pos':         (t_int_filt[0], t_int_filt[1], t_int_filt[2]),
                'pos_err':          trans_err_filt,
                'trans_errs':       tuple(trans_errs),
                'pred_ori':         (q[0], q[1], q[2]),
                'ori_err':          rot_err_filt,
                'axis_errs':        tuple(rot_errs),
                'keypoint_err':     mpjpe,
                
                # 2) Timestamp
                'sim_time':         sim_time,
                 
                 # 3) For drawing
                'keypoints':        predicted_2d,
                'rvec':             rvec,
                'tvec':             tvec.reshape(3,1),
                'camera_matrix':    cam.K,
                'dist_coeffs':      np.array(cam.dcoef, dtype=np.float32),
                 
                # 7) Misc stats
                'image_skips':      skip_count,
                'total_keypoints':  total_keypoints,
                'total_dropped':    total_dropped,
                'frames_dropped': drop_count,
                'frame_num' : frame_idx
                }


            # ——— 6a) Log orientation.csv ———
            # timestamp, frame num, pred_rx,pred_ry,pred_rz, gt_rx,gt_ry,gt_rz, total_pose_error
            data_logger.log_orientation(
                sim_time,
                frame_idx,
                (q[0], q[1], q[2]),
                (qx,   qy,   qz),
                rot_err_filt,
                E_pose
            )

            # ——— 6b) Log position.csv ———
            # timestamp, frame num, pred_tx,pred_ty,pred_tz, gt_tx,gt_ty,gt_tz, total_trans_error
            data_logger.log_position(
                sim_time,
                frame_idx,
                (t_int_filt[0], t_int_filt[1], t_int_filt[2]),
                (px, py, pz),
                trans_err_filt
            )

            # ——— 6c) Log keypoints.csv ———
            # timestamp, frame num, keypoint_error, total_keypoints, dropped_keypoints, frames_skipped
            data_logger.log_keypoints(
                sim_time,
                frame_idx,
                mpjpe,
                total_keypoints,
                dropped,
                skip_count
                )
                
            # ——— 6d) Log kalman_trans.csv ———
            # timestamp, frame num, keypoint_error, total_keypoints, dropped_keypoints, frames_skipped
            data_logger.log_kalman_trans(
                sim_time,
                frame_idx,
                trans_err,
                trans_err_filt
                )
                
            # ——— 6d) Log kalman_rot.csv ———
            # timestamp, frame num, keypoint_error, total_keypoints, dropped_keypoints, frames_skipped
            data_logger.log_kalman_rot(
                sim_time,
                frame_idx,
                rot_err,
                rot_err_filt
                )

            # drop old if necessary:
            if vis_queue.full():
                try: vis_queue.get_nowait()
                except: pass
            vis_queue.put_nowait(payload)
            
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user — shutting down cleanly.")
        
    finally:
        csv_file.close()
        conn.close()
        srv.close()
        data_logger.close()

    # final summary (unchanged)
    print("Total images skipped:", skip_count)
    print("Total frames dropped  (network/PNG errors):", drop_count)
    print(f"\n=== Keypoint drop summary ===")
    print(f"Total keypoints processed: {total_keypoints}")
    print(f"Total keypoints dropped : {total_dropped}")
    print(f"Overall drop rate        : {100 * total_dropped/total_keypoints:.2f}%")

if __name__ == "__main__":
    main()
