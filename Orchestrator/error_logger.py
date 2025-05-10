# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O’Brien (extended)
# ------------------------------------------------------------------------------
"""
error_logger.py

Provides ErrorLogger to accumulate per-frame translation and attitude errors
and save them to CSV files.
"""
import pandas as pd

class ErrorLogger:
    """
    Accumulates translation and attitude error records in memory, then writes
    them out as CSV files when requested.
    """
    def __init__(self):
        """
        Initialize empty lists for translation and attitude error rows.
        """
        self._trans_rows = []
        self._att_rows   = []

    def log_translation(self, timestamp, frame, x_err, y_err, z_err):
        """
        Append a translation-error record for a single frame.

        Args:
            timestamp (float): Simulation time in seconds.
            frame (int): Frame index.
            x_err (float): Error along the X axis (meters).
            y_err (float): Error along the Y axis (meters).
            z_err (float): Error along the Z axis (meters).
        """
        self._trans_rows.append({
            'timestamp': timestamp,
            'frame_num': frame,
            'x_error':   x_err,
            'y_error':   y_err,
            'z_error':   z_err,
        })

    def log_attitude(self, timestamp, frame, roll_err, yaw_err, pitch_err):
        """
        Append an attitude-error record for a single frame.

        Args:
            timestamp (float): Simulation time in seconds.
            frame (int): Frame index.
            roll_err (float): Roll error (radians).
            yaw_err (float): Yaw error (radians).
            pitch_err (float): Pitch error (radians or degrees, depending on context).
        """
        self._att_rows.append({
            'timestamp':    timestamp,
            'frame_num':    frame,
            'roll_error':   roll_err,
            'yaw_error':    yaw_err,
            'pitch_error':  pitch_err,
        })

    def save(self,
             trans_filename: str = 'translation_errors.csv',
             att_filename:   str = 'attitude_errors.csv'):
        """
        Write accumulated error logs to CSV files.

        Args:
            trans_filename (str): Path for the translation-errors CSV.
            att_filename (str): Path for the attitude-errors CSV.
        """
        # Build DataFrames
        df_t = pd.DataFrame(self._trans_rows)
        df_a = pd.DataFrame(self._att_rows)

        # Write CSV directly
        df_t.to_csv(trans_filename, index=False)
        df_a.to_csv(att_filename, index=False)

        print(f"Wrote translation log → {trans_filename}")
        print(f"Wrote attitude   log → {att_filename}")
