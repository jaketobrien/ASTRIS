# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O'Brien
# ------------------------------------------------------------------------------

"""
vis.py

Overview:
    Background, low-priority visualization for real-time inference.
    Shows overall and per-axis translation & rotation errors, keypoint error,
    chaser/target speeds, and live frames with colored X/Y/Z axes.
    Optionally records the visualization to a video file.

Usage:
    from multiprocessing import Queue
    from vis import start_visualizer

    q = Queue(maxsize=2)
    viz_proc = start_visualizer(q)

    # In your main loop, push data dicts into `q`:
    q.put({
        'frame': frame,
        'pred_pos': (x, y, z),
        'pos_err': total_pos_err,
        'trans_errs': (ex, ey, ez),
        'pred_ori': (rx, ry, rz),
        'ori_err': total_ori_err,
        'axis_errs': (erx, ery, erz),
        'keypoint_err': kp_err,
        'sim_time': timestamp,
        'keypoints': keypoints_array,
        'rvec': rvec,
        'tvec': tvec,
        'camera_matrix': cam.K,
        'dist_coeffs': cam.dcoef,
        'image_skips': skips,
        'total_keypoints': total_kpts,
        'total_dropped': dropped_kpts,
        'frames_dropped': dropped_frames,
        'frame_num': frame_index
    })

    # When done, terminate:
    viz_proc.terminate()

Outputs:
    • An OpenCV window titled “ASTRIS – Mission Control” showing stats + live feed  
    • Optionally, an MP4 video file if `Visualizer(out_vid=…)` is provided  
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import time
from multiprocessing import Process, Queue
from queue import Empty

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import cv2
import numpy as np

# Draw keypoints & skeleton lines
def draw_keypoints_and_lines(img, points, pairs,
                              point_color=(0,165,255), line_color=(0,165,255),
                              radius=5, thickness=2):
    """
    Draw 2D keypoints and skeleton connections on an image.

    Args:
        img: BGR image array of shape (H, W, 3).
        points: Array of shape (N, 2) containing (x, y) coordinates.
        pairs: List of index pairs defining skeleton edges.
        point_color: BGR color tuple for drawing keypoints.
        line_color: BGR color tuple for drawing skeleton lines.
        radius: Radius of each keypoint circle.
        thickness: Line thickness for skeleton connections.

    Returns:
        Annotated image with keypoints and skeleton lines.
    """
    for x, y in points.astype(int):
        cv2.circle(img, (x, y), radius, point_color, -1)
    for i, j in pairs:
        x1, y1 = points[i].astype(int)
        x2, y2 = points[j].astype(int)
        cv2.line(img, (x1, y1), (x2, y2), line_color, thickness)
    return img

# Draw RGB axes
def draw_axes(img, rvec, tvec, camera_matrix, dist_coeffs,
              length=1.0, thickness=2, scale_factor=0.5):
    """
    Project and draw 3D coordinate axes on an image.

    Args:
        img: BGR image array of shape (H, W, 3).
        rvec: Rodrigues rotation vector (3,1).
        tvec: Translation vector (3,1).
        camera_matrix: Intrinsic camera matrix (3x3).
        dist_coeffs: Distortion coefficients array.
        length: Physical length of each axis in world units.
        thickness: Thickness of the drawn axis lines.
        scale_factor: Fraction of full axis length to render.

    Returns:
        Annotated image with projected RGB axes.
    """
    colors = {'X': (0, 0, 255), 'Y': (0, 255, 0), 'Z': (255, 0, 0)}
    pts3d = np.float32([[0,0,0],
                        [length,0,0],
                        [0,length,0],
                        [0,0,length]])
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts2d = pts2d.reshape(-1,2).astype(int)
    origin = tuple(pts2d[0])
    for idx, axis in enumerate(('X','Y','Z'), start=1):
        endpoint = tuple((pts2d[0] + (pts2d[idx] - pts2d[0]) * scale_factor).astype(int))
        cv2.line(img, origin, endpoint, colors[axis], thickness)
        cv2.putText(img, axis, endpoint, cv2.FONT_HERSHEY_DUPLEX, 1.0, colors[axis], 2)
    return img

PAIRS = [
    (0,1),(1,2),(2,3),(0,3),
    (4,5),(5,6),(6,7),(4,7),
    (4,9),(6,8),(5,10),(0,7),
    (1,4),(2,5),(3,6)
]

class Visualizer:
    """
    Real-time visualization panel for displaying inference metrics and video.

    Args:
        window_name: Name of the OpenCV display window.
        panel_width: Width of the statistics panel (pixels).
        fps: Frame rate for display refresh.
        out_vid: Optional path to save recorded video.
        record_fps: Frame rate for video recording (defaults to fps).
    """
    def __init__(self, window_name='ASTRIS - Mission Control', panel_width=450, fps=30, out_vid=None, record_fps=None):
        self.win = window_name
        self.panel_w = panel_width
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.start_time = time.time()
        self.out_vid = out_vid       # e.g. "session.mp4"
        self.record_fps = record_fps or fps
        self.writer = None          # will be cv2.VideoWriter once initialized
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self.bg_color = (0,0,0)
        self.text_color = (255,255,255)
        self.value_color = (0,165,255)
        self.last_canvas = None

    def update(self,
        frame,
        pred_pos, pos_err, trans_errs,
        pred_ori, ori_err, axis_errs,
        keypoint_err,
        sim_time,
        keypoints, rvec, tvec,
        camera_matrix, dist_coeffs,
        image_skips, total_keypoints, total_dropped, frames_dropped, frame_num):
        """
        Update the visualization window with the latest frame and metrics.

        Args:
            frame: BGR image frame (HxWx3).
            pred_pos: Predicted translation (x, y, z).
            pos_err: Total translation error.
            trans_errs: Per-axis translation errors.
            pred_ori: Predicted orientation (Euler angles).
            ori_err: Total rotation error.
            axis_errs: Per-axis rotation errors.
            keypoint_err: Keypoint reprojection error (pixels).
            sim_time: Simulation timestamp.
            keypoints: Array of 2D keypoints (Nx2).
            rvec: Rodrigues rotation vector for pose.
            tvec: Translation vector for pose.
            camera_matrix: Intrinsic camera matrix.
            dist_coeffs: Distortion coefficients.
            image_skips: Number of skipped frames.
            total_keypoints: Total keypoints processed.
            total_dropped: Total keypoints dropped.
            frames_dropped: Total frames dropped.
            frame_num: Current frame index.
        """

        h, w = frame.shape[:2]
        canvas = np.zeros((h, self.panel_w + w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # Text layout
        lh = 28
        x0, y = 10, 30
        
        def put(label, val, y):
            cv2.putText(canvas, label, (x0, y), self.font, 0.45, self.text_color, 1)
            cv2.putText(canvas, val,   (x0+240, y), self.font, 0.45, self.value_color, 1)

        # Translation
        put("Position (m):",            f"{pred_pos[0]:.2f},{pred_pos[1]:6.2f},{pred_pos[2]:6.2f}", y); y += lh
        put("X,Y,Z Error (m):",         f"{trans_errs[0]:.2f},{trans_errs[1]:6.2f}, {trans_errs[2]:6.2f}", y); y += lh
        put("Total Position Error (m):",f"{pos_err:.2f}", y); y += lh + 10

        # Rotation
        put("Orientation (deg):",       f"{pred_ori[0]:.2f},{pred_ori[1]:6.2f},{pred_ori[2]:6.2f}", y); y += lh
        put("X,Y,Z Error (deg):",       f"{axis_errs[0]:.2f}, {axis_errs[1]:6.2f}, {axis_errs[2]:6.2f}",  y); y += lh
        put("Total Orientation Error:", f"{ori_err:.2f}", y); y += lh + 10
        
        # Keypoint & stats
        put("Keypoint Error (px):",     f"{keypoint_err:.2f}", y); y += lh
        put("Total Keypoints:",         f"{total_keypoints}",   y); y += lh
        put("Keypoints Dropped:",       f"{total_dropped}",     y); y += lh
        put("Dropped (%):",             f"{100 * total_dropped/total_keypoints:.2f}%", y); y += lh
        put("Frames Skipped:",          f"{image_skips}",       y); y += lh + 10
        
        put("Frame:",                   f"{frame_num}",         y); y += lh
        put("Timestamp:",               f"{sim_time}",          y); y += lh
        put("Frames Dropped:",          f"{frames_dropped}",    y); y += lh + 40

        # Timer
        elapsed = int(time.time() - self.start_time)
        mm, ss = divmod(elapsed, 60)

        cv2.putText(canvas, f"Time {mm:02d}:{ss:02d}", (x0, h - 10),
                    self.font, 0.6, self.value_color, 1)

        # draw frame with keypoints + axes
        vis = frame.copy()
        draw_keypoints_and_lines(vis, keypoints, PAIRS,
                                 point_color=self.value_color,
                                 line_color=self.value_color)
        draw_axes(vis, rvec, tvec, camera_matrix, dist_coeffs)
        canvas[:, self.panel_w:] = vis

        # Initialize video writer once
        if self.out_vid and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.out_vid,
                fourcc,
                self.record_fps,
                (canvas.shape[1], canvas.shape[0]),
                True
            )

        # Write full canvas to video
        if self.writer:
            self.writer.write(canvas)

        # Show on screen
        cv2.imshow(self.win, canvas)
        cv2.waitKey(int(1000 / self.fps))
        
        self.last_canvas = canvas.copy() # Stash for timeout

    def close(self):
        """
        Release video writer (if any) and destroy all OpenCV windows.
        """
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

def viz_worker(q: Queue):
    """
    Background worker that reads data dicts and updates the Visualizer.

    Args:
        q: Multiprocessing queue providing visualization data.
    """
    # Pass a filename to start saving the video
    vis = Visualizer(out_vid="LEO.mp4", record_fps=1)
    last_frame_time = time.time()
    try:
        while True:
            try:
                # Wait up to one frame-time for data
                data = q.get(timeout=1/vis.fps)
                # Got a new frame, reset timer
                last_frame_time = time.time()
                vis.update(
                    data['frame'],
                    data['pred_pos'], data['pos_err'], data['trans_errs'],
                    data['pred_ori'], data['ori_err'], data['axis_errs'],
                    data['keypoint_err'], data['sim_time'],
                    data['keypoints'], data['rvec'], data['tvec'],
                    data['camera_matrix'], data['dist_coeffs'],
                    data['image_skips'], data['total_keypoints'],
                    data['total_dropped'], data['frames_dropped'],
                    data['frame_num'])
            except Empty:
                # No frame this cycle, check how long it’s been
                idle = time.time() - last_frame_time
                if idle > 15.0 and vis.last_canvas is not None:
                # Draw the two lines on the cached canvas
                    canvas = vis.last_canvas.copy()
                    x0, y = 10, 30
                    lh = 28
                    y += lh + 450
                    #cv2.putText(canvas, "DOCKING",   (x0, y),       vis.font, 2.0, (0,165,255), 3)
                    y += lh + 40
                    #cv2.putText(canvas, "INITIATED", (x0, y),       vis.font, 2.0, (0,165,255), 3)
                    cv2.imshow(vis.win, canvas)
                    cv2.waitKey(1)
                    # Reset timer so it only draws once
                    last_frame_time = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        vis.close()

def start_visualizer(queue: Queue) -> Process:
    """
    Spawn a daemon process for real-time visualization.

    Args:
        queue: Multiprocessing queue for passing frames and metrics.

    Returns:
        The spawned Process object.
    """
    p = Process(target=viz_worker, args=(queue,), daemon=True)
    p.start()
    return p

