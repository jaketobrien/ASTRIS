# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O'Brien
# ------------------------------------------------------------------------------

"""
log.py

Overview:
A simple CSV logger for pose-estimation metrics.
Records per-frame orientation, position, keypoint, and Kalman diagnostics.

Usage:
from log import Logger
logger = Logger(output_dir="results_log")
logger.log_orientation(timestamp, frame_num, pred_rot, gt_rot, rot_err, pose_err)
logger.log_position(timestamp, frame_num, pred_trans, gt_trans, trans_err)
logger.log_keypoints(timestamp, frame_num, kp_err, total_kpts, dropped_kpts, skipped_frames)
logger.log_kalman_trans(timestamp, frame_num, trans_err, trans_err_filt)
logger.log_kalman_rot(timestamp, frame_num, rot_err, rot_err_filt)
logger.close()

Outputs:
• orientation.csv       — timestamp, frame, predicted vs. GT rotations, rotation & pose errors
• position.csv          — timestamp, frame, predicted vs. GT translations, translation errors
• keypoints.csv         — timestamp, frame, MPJPE, total/dropped keypoints, skipped frames
• kalman_trans.csv      — timestamp, frame, raw vs. filtered translation errors
• kalman_rot.csv        — timestamp, frame, raw vs. filtered rotation errors
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import csv
import os
from multiprocessing import current_process


class Logger:
    """
    Simple CSV logger for pose-estimation diagnostics.

    Creates and manages five CSV files in the given output directory:
      - orientation.csv      : per-frame predicted vs ground-truth rotations and pose error
      - position.csv         : per-frame predicted vs ground-truth translations and translation error
      - keypoints.csv        : per-frame keypoint error, counts, and skips
      - kalman_trans.csv     : per-frame raw vs filtered translation errors
      - kalman_rot.csv       : per-frame raw vs filtered rotation errors

    Each CSV is initialized with a header row upon construction.
    """
    def __init__(self, output_dir='.'):
        """
        Initialize the Logger by creating CSV files and writing headers.

        Args:
            output_dir: Directory where CSV log files will be created.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # File paths
        self.orient_path = os.path.join(output_dir, 'orientation.csv')
        self.pos_path    = os.path.join(output_dir, 'position.csv')
        self.kp_path     = os.path.join(output_dir, 'keypoints.csv')
        self.kalman_trans_path  = os.path.join(output_dir, 'kalman_trans.csv')
        self.kalman_rot_path  = os.path.join(output_dir, 'kalman_rot.csv')

        # Open files for writing
        self.orient_file = open(self.orient_path, 'w', newline='')
        self.pos_file    = open(self.pos_path,    'w', newline='')
        self.kp_file     = open(self.kp_path,     'w', newline='')
        self.kalman_trans_file = open(self.kalman_trans_path, 'w', newline='')
        self.kalman_rot_file = open(self.kalman_rot_path, 'w', newline='')

        # CSV writers
        self.orient_writer = csv.writer(self.orient_file)
        self.pos_writer    = csv.writer(self.pos_file)
        self.kp_writer     = csv.writer(self.kp_file)
        self.kalman_trans_writer = csv.writer(self.kalman_trans_file)
        self.kalman_rot_writer = csv.writer(self.kalman_rot_file)

        # Write headers
        self.orient_writer.writerow([
            'timestamp', 'frame_num',
            'pred_rot_x', 'pred_rot_y', 'pred_rot_z',
            'gt_rot_x',   'gt_rot_y',   'gt_rot_z',
            'total_rot_error', 'total_pose_error'
        ])
        self.pos_writer.writerow([
            'timestamp', 'frame_num',
            'pred_trans_x', 'pred_trans_y', 'pred_trans_z',
            'gt_trans_x',   'gt_trans_y',   'gt_trans_z',
            'total_trans_error'
        ])
        self.kp_writer.writerow([
            'timestamp', 'frame_num',
            'keypoint_error', 'total_keypoints',
            'dropped_keypoints', 'frames_skipped'
        ])
        self.kalman_trans_writer.writerow([
            'timestamp', 'frame_num',
            'trans_err', 'trans_err_filt'
        ])
        self.kalman_rot_writer.writerow([
            'timestamp', 'frame_num',
            'rot_err', 'rot_err_filt'
        ])

    def log_orientation(self, timestamp, frame_num, pred_rot, gt_rot, total_rot_error, total_pose_error):
        """
        Log orientation and combined pose error metrics for a single frame.

        Args:
            timestamp: Simulation time or frame timestamp.
            frame_num: Index of the frame.
            pred_rot: Predicted rotation Euler angles (rx, ry, rz).
            gt_rot: Ground-truth rotation Euler angles (rx, ry, rz).
            total_rot_error: Scalar rotation error (degrees).
            total_pose_error: Combined pose error metric.
        """
        self.orient_writer.writerow([
            timestamp,
            frame_num,
            pred_rot[0], pred_rot[1], pred_rot[2],
            gt_rot[0],   gt_rot[1],   gt_rot[2],
            total_rot_error,
            total_pose_error
        ])

    def log_position(self, timestamp, frame_num, pred_trans, gt_trans, total_trans_error):
        """
        Log translation error metrics for a single frame.

        Args:
            timestamp: Simulation time or frame timestamp.
            frame_num: Index of the frame.
            pred_trans: Predicted translation vector (tx, ty, tz).
            gt_trans: Ground-truth translation vector (tx, ty, tz).
            total_trans_error: Scalar translation error.
        """
        self.pos_writer.writerow([
            timestamp,
            frame_num,
            pred_trans[0], pred_trans[1], pred_trans[2],
            gt_trans[0],   gt_trans[1],   gt_trans[2],
            total_trans_error
        ])

    def log_keypoints(self, timestamp, frame_num, keypoint_error, total_keypoints, dropped_keypoints, frames_skipped):
        """
        Log 2D keypoint error statistics for a single frame.

        Args:
            timestamp: Simulation time or frame timestamp.
            frame_num: Index of the frame.
            keypoint_error: Mean per-joint pixel error.
            total_keypoints: Number of keypoints processed.
            dropped_keypoints: Number dropped by RANSAC.
            frames_skipped: Number of frames skipped.
        """
        self.kp_writer.writerow([
            timestamp,
            frame_num,
            keypoint_error,
            total_keypoints,
            dropped_keypoints,
            frames_skipped
        ])
        
    def log_kalman_trans(self, timestamp, frame_num, trans_err, trans_err_filt):
        """
        Log raw vs. filtered translation error for Kalman diagnostics.

        Args:
            timestamp: Simulation time or frame timestamp.
            frame_num: Index of the frame.
            trans_err: Raw translation error.
            trans_err_filt: Filtered translation error.
        """
        self.kalman_trans_writer.writerow([
            timestamp,
            frame_num,
            trans_err, 
            trans_err_filt
        ])
        
    def log_kalman_rot(self, timestamp, frame_num, rot_err, rot_err_filt):
        """
        Log raw vs. filtered rotation error for Kalman diagnostics.

        Args:
            timestamp: Simulation time or frame timestamp.
            frame_num: Index of the frame.
            rot_err: Raw rotation error (degrees).
            rot_err_filt: Filtered rotation error (degrees).
        """
        self.kalman_rot_writer.writerow([
            timestamp,
            frame_num,
            rot_err, 
            rot_err_filt
        ])

    def close(self):
        """
        Close all file handles.
        """
        self.orient_file.close()
        self.pos_file.close()
        self.kp_file.close()
        self.kalman_trans_file.close()
        self.kalman_rot_file.close()

def logger_worker(log_queue, output_dir):
    """
    Background worker that consumes logging commands from a queue.

    Args:
        log_queue: Queue of (method_name, args) tuples.
        output_dir: Directory where Logger writes its CSV files.

    Behavior:
        - Instantiates a Logger.
        - Applies each (method, args) from the queue.
        - Stops, then closes the Logger when it receives None.
    """
    logger = Logger(output_dir=output_dir)
    while True:
        record = log_queue.get()
        if record is None:       # sentinel → shut down
            break
        method_name, args = record
        getattr(logger, method_name)(*args)
    logger.close()
