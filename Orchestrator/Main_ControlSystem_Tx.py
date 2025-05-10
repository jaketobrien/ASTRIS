# ------------------------------------------------------------------------------
# Copyright (c) 2025 Jake O’Brien
# Licensed under the Apache 2.0 License.
# Written by Jake O’Brien (extended)
# ------------------------------------------------------------------------------

"""
Main_ControlSystem_Tx.py

Overview:
    Sends video and pose frames while running a two-stage rendezvous control in parallel:
      1. Near-range PID hold from 10 m down to 3 m (stage 1).
      2. Terminal Range full 6-DOF translation + attitude control on a 640×640 feed (stage 2).
    Automatically initiates docking exactly two minutes after entering the TR phase.

Usage:
    python Main_ControlSystem_Tx.py

Configuration:
    Adjust the JETSON_IP, JETSON_PORT, and FPS constants at the top of the script
    to match the receiver network and frame-rate requirements.

Outputs:
    • Streaming of timestamped pose + PNG-encoded frames over TCP to the receiver.
    • Console logs for control actions and status updates.
    • Two CSV files recording control errors:
        – translation_errors.csv
        – attitude_errors.csv
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import os
import sys
import time
import socket
import struct
import threading
import logging
import math

# ------------------------------------------------------------------------------
# Ensure this script’s directory is first on the module search path
# ------------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import cv2
import numpy as np

# ------------------------------------------------------------------------------
# Local application imports
# ------------------------------------------------------------------------------
from error_logger import ErrorLogger

# ------------------------------------------------------------------------------
# MDS-client imports
# ------------------------------------------------------------------------------
sys.path.insert(0, r"C:\…\mds-python\src")
from mds_client import MdsClient
from mds_client.generated.agent.thrusters_pb2 import ThrustState
from mds_client.generated.agent.agent_overrides_pb2 import TorqueRequest
from mds_client.generated import general_pb2
from mds_client.generated.agent import cameras_pb2

# CV parameters
JETSON_IP   = "192.168.1.100"
JETSON_PORT = 5000
FPS         = 1.0

# Global Variables
tr_started = False

#---------------------------------------------
# Shared Helpers (from FullTR.py)
#---------------------------------------------
def setup_video_feed(client, image_size):
    """
    Configure and start a video feed on the MDS client.

    Args:
        client: An MdsClient instance.
        image_size: Desired square frame size (pixels).

    Returns:
        feed: A FeedHandle for the started video stream.
    """
    reply, feeds = client.get_video_feeds()
    if reply.status == general_pb2.BaseResponse.ResponseCode.OK and feeds:
        feed = feeds[0]
    else:
        feed = cameras_pb2.FeedHandle(agentId="Chaser", feedId="main")
    client.set_feed_size(feed, image_size, image_size)
    client.set_feed_fov(feed, 60)
    client.set_feed_buffer_size(3)
    client.start_video_feed(feed)
    time.sleep(0.2)
    return feed

def get_translation_component(client, feed, axis):
    """
    Retrieve a single axis of the relative translation to the target.

    Args:
        client: An MdsClient instance.
        feed: The FeedHandle used for relative pose queries.
        axis: One of 'x', 'y', or 'z'.

    Returns:
        float or None: Absolute distance along that axis, or None if unavailable.
    """
    _, relPos = client.get_relative_to_camera_position("TANGO", feed)
    if relPos and hasattr(relPos, 'position'):
        if axis == 'x': return relPos.position.x
        if axis == 'y': return relPos.position.y
        if axis == 'z': return abs(relPos.position.z)
    return None

def wrap_angle(x):
    """
    Wrap an angle in radians to the range [-π, +π].

    Args:
        x: Angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    return (x + math.pi) % (2*math.pi) - math.pi

def quat_to_yaw(x, y, z, w):
    """
    Compute yaw (around Z) from a quaternion.

    Args:
        x, y, z, w: Quaternion components.

    Returns:
        Yaw angle in radians.
    """
    return math.atan2(2*(w*z + x*y),
                      1 - 2*(y*y + z*z))

def quat_to_pitch(x, y, z, w):
    """
    Compute pitch (around Y) from a quaternion.

    Args:
        x, y, z, w: Quaternion components.

    Returns:
        Pitch angle in radians.
    """
    return math.atan2(2*(w*y + x*z),
                      1 - 2*(y*y + z*z))

def quat_conjugate(q):
    """
    Compute the conjugate of a quaternion.

    Args:
        q: Quaternion as [x, y, z, w].

    Returns:
        Conjugated quaternion [−x, −y, −z, w].
    """
    return [-q[0], -q[1], -q[2], q[3]]

def quat_multiply(a, b):
    """
    Multiply two quaternions a * b.

    Args:
        a, b: Quaternions as [x, y, z, w].

    Returns:
        Product quaternion [x, y, z, w].
    """
    x1,y1,z1,w1 = a
    x2,y2,z2,w2 = b
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def log_frame_errors(sim_time, frame, pos, rot, desired_q, logger):
    """
    Compute and log translation and attitude errors for a single frame.

    Args:
        sim_time: Simulation timestamp (s).
        frame: Frame index (int).
        pos: Position object with .x, .y, .z.
        rot: Rotation object with .x, .y, .z, .w.
        desired_q: Reference quaternion [x, y, z, w].
        logger: ErrorLogger instance for writing CSVs.
    """
    # --- translation errors ---
    x_err = pos.x
    y_err = pos.y
    z_err = abs(pos.z) - 3.0

    # --- attitude errors ---
    qmeas = [rot.x, rot.y, rot.z, rot.w]
    qe    = quat_multiply(qmeas, quat_conjugate(desired_q))
    if qe[3] < 0:
        qe = [-c for c in qe]
    e_vec    = [2*qe[0], 2*qe[1], 2*qe[2]]

    roll_err = -e_vec[2]                     # leave roll in radians
    yaw_err  = wrap_angle(math.atan2(pos.x, pos.z))  # yaw in radians

    # only pitch converted to degrees:
    pitch_err = math.degrees(-e_vec[1])

    # log them
    logger.log_translation(sim_time, frame, x_err, y_err, z_err)
    logger.log_attitude(  sim_time, frame, roll_err, yaw_err, pitch_err)


#---------------------------------------------
# PID class (6-DOF)
#---------------------------------------------
class PID:
    """
    A PID controller with output limiting.
    """
    def __init__(self, Kp, Ki, Kd, limit):
        """
        Args:
            Kp, Ki, Kd: PID gains.
            limit: Absolute bound on the controller output.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.i = 0.0
        self.prev = 0.0
        self.limit = limit

    def __call__(self, err, dt):
        """
        Compute control output for a given error and timestep.

        Args:
            err: Current error.
            dt: Time since last update.

        Returns:
            Clamped PID output.
        """
        self.i += err * dt
        d = (err - self.prev) / dt if dt > 0 else 0.0
        self.prev = err
        u = self.Kp*err + self.Ki*self.i + self.Kd*d
        return max(-self.limit, min(self.limit, u))

#---------------------------------------------
# Stage 1 PID loops
#---------------------------------------------
def pid_control_loop_z_NR(client, feed, desired_distance, update_interval, stop_event):
    """
    Hold Z at a fixed distance (NR phase) using simple PID until setpoint reached.

    Args:
        client: MdsClient instance.
        feed: Video feed handle.
        desired_distance: Target Z-distance (m).
        update_interval: Control loop period (s).
        stop_event: threading.Event to signal loop exit.
    """
    Kp, Ki, Kd = 0.0001, 0.000001, 2.5
    integral = prev = 0.0
    max_p, min_p = 0.3, 0.0
    frame = 0
    logging.info("Z-loop: hold %.2f m until threshold", desired_distance)
    while not stop_event.is_set():
        t0 = time.time(); frame += 1
        dist = get_translation_component(client, feed, 'z')
        if dist is None:
            time.sleep(update_interval); continue

        err = dist - desired_distance
        integral += err * update_interval
        deriv = (err - prev) / update_interval
        prev = err
        u = Kp*err + Ki*integral + Kd*deriv
        pulse = min(max(abs(u), min_p), max_p)

        logging.info("Z %3d: dist=%.2f err=%.2f ctrl=%.2f pulse=%.2f",
                     frame, dist, err, u, pulse)
        thr = "ThrusterForward" if u>0 else "ThrusterBack"
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
        time.sleep(pulse)
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

        if dist <= desired_distance:
            logging.info("Reached %.2f m — end Stage 1", desired_distance)
            stop_event.set()
            break

        dt = time.time() - t0
        if dt < update_interval:
            time.sleep(update_interval - dt)

def pid_control_loop_z_TR(client, feed, desired_distance, update_interval, stop_event):
    """
    Hold Z at a fixed distance (TR stage 1) using tuned PID until setpoint reached.
    """
    Kp, Ki, Kd = 0.05, 0.00001, 1.6
    integral = prev = 0.0
    max_p, min_p = 0.3, 0.0
    frame = 0
    logging.info("Z-loop: hold %.2f m until threshold", desired_distance)
    while not stop_event.is_set():
        t0 = time.time(); frame += 1
        dist = get_translation_component(client, feed, 'z')
        if dist is None:
            time.sleep(update_interval); continue

        err = dist - desired_distance
        integral += err * update_interval
        deriv = (err - prev) / update_interval
        prev = err
        u = Kp*err + Ki*integral + Kd*deriv
        pulse = min(max(abs(u), min_p), max_p)

        logging.info("Z %3d: dist=%.2f err=%.2f ctrl=%.2f pulse=%.2f",
                     frame, dist, err, u, pulse)
        thr = "ThrusterForward" if u>0 else "ThrusterBack"
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
        time.sleep(pulse)
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

        if dist <= desired_distance:
            logging.info("Reached %.2f m — end Stage 1", desired_distance)
            stop_event.set()
            break

        dt = time.time() - t0
        if dt < update_interval:
            time.sleep(update_interval - dt)

def pid_control_loop_x(client, feed, desired, update_interval, stop_event):
    """
    Center X to setpoint (m) using PID in parallel with other loops.
    """
    Kp, Ki, Kd = 0.05, 0.0, 1.6
    integral = prev = 0.0
    max_p, min_p = 1.0, 0.0
    frame = 0
    logging.info("X-loop: center at %.2f m", desired)
    while not stop_event.is_set():
        t0 = time.time(); frame += 1
        x = get_translation_component(client, feed, 'x')
        if x is None:
            time.sleep(update_interval); continue

        err = x - desired
        integral += err * update_interval
        deriv = (err - prev) / update_interval
        prev = err
        u = Kp*err + Ki*integral + Kd*deriv
        pulse = min(max(abs(u), min_p), max_p)

        logging.info("X %3d: lat=%.2f err=%.2f ctrl=%.2f pulse=%.2f",
                     frame, x, err, u, pulse)
        thr = "ThrusterRight" if u>0 else "ThrusterLeft"
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
        time.sleep(pulse)
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

        dt = time.time() - t0
        if dt < update_interval:
            time.sleep(update_interval - dt)

def pid_control_loop_y(client, feed, desired, update_interval, stop_event):
    """
    Center Y to setpoint (m) using PID in parallel with other loops.
    """
    Kp, Ki, Kd = 0.05, 0.0, 1.6
    integral = prev = 0.0
    max_p, min_p = 1.0, 0.0
    frame = 0
    logging.info("Y-loop: center at %.2f m", desired)
    while not stop_event.is_set():
        t0 = time.time(); frame += 1
        y = get_translation_component(client, feed, 'y')
        if y is None:
            time.sleep(update_interval); continue

        err = y - desired
        integral += err * update_interval
        deriv = (err - prev) / update_interval
        prev = err
        u = Kp*err + Ki*integral + Kd*deriv
        pulse = min(max(abs(u), min_p), max_p)

        logging.info("Y %3d: vert=%.2f err=%.2f ctrl=%.2f pulse=%.2f",
                     frame, y, err, u, pulse)
        thr = "ThrusterUp" if u>0 else "ThrusterDown"
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
        time.sleep(pulse)
        client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

        dt = time.time() - t0
        if dt < update_interval:
            time.sleep(update_interval - dt)

#---------------------------------------------
# Stage 2: 6-DOF loops
#---------------------------------------------
def translation_loop_6dof(client, feed, pid_x, pid_y, pid_z):
    """
    Continuously correct X, Y, Z to 0/3 m setpoints using three PID controllers.

    Args:
        client: MdsClient instance.
        feed: Video feed handle.
        pid_x, pid_y, pid_z: PID instances for each axis.
    """
    interval = 1.0
    logging.info("Stage 2 Translation loop started.")
    try:
        while True:
            t0 = time.time()
            _, rel = client.get_relative_to_camera_position("TANGO", feed)
            if not rel or not hasattr(rel, 'position'):
                time.sleep(interval); continue

            # X
            cx = rel.position.x
            ux = pid_x(cx - 0.0, interval)
            thr = "ThrusterRight" if ux>0 else "ThrusterLeft"
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
            time.sleep(abs(ux))
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

            # Y
            cy = rel.position.y
            uy = pid_y(cy - 0.0, interval)
            thr = "ThrusterUp" if uy>0 else "ThrusterDown"
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
            time.sleep(abs(uy))
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

            # Z
            cz = abs(rel.position.z)
            uz = pid_z(cz - 3.0, interval)
            thr = "ThrusterForward" if uz>0 else "ThrusterBack"
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=True)])
            time.sleep(abs(uz))
            client.set_thruster_states([ThrustState(agentId="Chaser", id=thr, isFiring=False)])

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    except KeyboardInterrupt:
        logging.info("Stage 2 Translation loop stopped.")

def attitude_loop(client, feed, pid_yaw, pid_pitch, pid_roll):
    """
    Maintain attitude using PID on yaw, pitch, and roll via torque requests.

    Args:
        client: MdsClient instance.
        feed: Video feed handle.
        pid_yaw, pid_pitch, pid_roll: PID instances for each rotation axis.
    """
    _, rel0 = client.get_relative_to_camera_position("TANGO", feed)
    q0 = rel0.rotation
    desired_q = [q0.x, q0.y, q0.z, q0.w]

    prev_t = time.time()
    dt_max = 0.5
    logging.info("Stage 2 Attitude loop started.")
    try:
        while True:
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt <= 0 or dt > dt_max:
                continue

            _, rel = client.get_relative_to_camera_position("TANGO", feed)
            if not rel:
                time.sleep(0.01); continue

            # Yaw
            yaw_meas = quat_to_yaw(rel.rotation.x, rel.rotation.y,
                                   rel.rotation.z, rel.rotation.w)
            bearing = wrap_angle(math.atan2(rel.position.x,
                                            rel.position.z))
            err_yaw = wrap_angle(bearing)
            u_yaw = -pid_yaw(err_yaw, dt)

            # Roll & Pitch via quaternion error
            qm = rel.rotation
            qmeas = [qm.x, qm.y, qm.z, qm.w]
            qe = quat_multiply(qmeas, quat_conjugate(desired_q))
            if qe[3] < 0:
                qe = [-c for c in qe]
            e_vec = [2*qe[0], 2*qe[1], 2*qe[2]]
            err_roll  = -e_vec[2]
            err_pitch = -e_vec[1]
            u_roll    = pid_roll(err_roll, dt)
            u_pitch   = pid_pitch(err_pitch, dt)

            client.set_torque_modifier(TorqueRequest(
                agentId="Chaser",
                torque=general_pb2.Vector3dData(
                    x=u_pitch,    # pitch
                    y=u_roll,     # roll
                    z=u_yaw       # yaw
                )
            ))

            logging.info(
                "Att: yaw=%.3f pitch=%.3f roll=%.3f → u=(%.6f,%.6f,%.6f)",
                err_yaw, math.degrees(err_pitch), math.degrees(err_roll),
                u_yaw, u_pitch, u_roll
            )
    except KeyboardInterrupt:
        client.set_torque_modifier(TorqueRequest(
            agentId="Chaser",
            torque=general_pb2.Vector3dData(x=0, y=0, z=0)
        ))
        logging.info("Stage 2 Attitude loop stopped.")

#---------------------------------------------
# Orchestrator Thread
#---------------------------------------------
def control_orchestrator(client, docking_event):
    """
    Coordinate Stage 1 (NR & TR stage 1) and Stage 2 (6-DOF) control phases.

    Args:
        client: MdsClient instance.
        docking_event: threading.Event to signal when to initiate docking.
    """
    global tr_started
    image_size_nr = 1024
    image_size_tr = 640

    # === NR phase: run translation PIDs until 10 m ===
    fnr_z = setup_video_feed(client, image_size_nr)
    fnr_x = setup_video_feed(client, image_size_nr)
    fnr_y = setup_video_feed(client, image_size_nr)
    stop_evt_nr = threading.Event()

    # Z uses NR z‐PID down to 10 m, X/Y center at 0
    for fn, args in [
        (pid_control_loop_z_NR, (client, fnr_z, 10.0, 0.3, stop_evt_nr)),
        (pid_control_loop_x,    (client, fnr_x,  0.0, 1.0, stop_evt_nr)),
        (pid_control_loop_y,    (client, fnr_y,  0.0, 1.0, stop_evt_nr))
    ]:
        threading.Thread(target=fn, args=args, daemon=True).start()

    stop_evt_nr.wait()
    for f in (fnr_x, fnr_y, fnr_z):
        client.stop_video_feed(f)
    logging.info("NR phase complete (≤10 m) → switching to TR")

    # === TR phase Stage 1: z=3m + x/y centering on 640×640 ===
    fz = setup_video_feed(client, image_size_tr)
    tr_started = True
    fx = setup_video_feed(client, image_size_tr)
    fy = setup_video_feed(client, image_size_tr)
    stop_evt_tr = threading.Event()

    for fn, args in [
        (pid_control_loop_z_TR, (client, fz, 3.0, 0.3, stop_evt_tr)),
        (pid_control_loop_x,    (client, fx, 0.0, 1.0, stop_evt_tr)),
        (pid_control_loop_y,    (client, fy, 0.0, 1.0, stop_evt_tr))
    ]:
        threading.Thread(target=fn, args=args, daemon=True).start()

    stop_evt_tr.wait()
    for f in (fx, fy, fz):
        client.stop_video_feed(f)
    logging.info("TR Stage 1 complete (≤3 m) → launching 6-DOF")

    # === TR phase Stage 2: 6-DOF on 640×640 ===
    time.sleep(0.5)
    f6 = setup_video_feed(client, image_size_tr)
    pid_x2     = PID(1.0,    0.00001,    1.8,   limit=1.0)
    pid_y2     = PID(1.0,    0.00001,    1.8,   limit=1.0)
    pid_z2     = PID(0.15,   0.00001,    1.8,   limit=1.4)
    pid_yaw2   = PID(0.001,  0.00001,    0.01,  limit=0.1)
    pid_pitch2 = PID(0.0005, 0.0000001,  0.000001, limit=0.5)
    pid_roll2  = PID(0.01,   0.00001,    0.5,   limit=0.1)

    threading.Thread(target=translation_loop_6dof,
                     args=(client, f6, pid_x2, pid_y2, pid_z2),
                     daemon=True).start()
    threading.Thread(target=attitude_loop,
                     args=(client, f6, pid_yaw2, pid_pitch2, pid_roll2),
                     daemon=True).start()

    # Hand back to main() to handle docking
    return


#---------------------------------------------
# Main: CV loop + control thread + 2 min timer
#---------------------------------------------
def main():
    """
    Entry point: streams timestamped pose+frames over TCP, runs control threads,
    and triggers docking after two minutes of TR phase.

    Usage:
        python Main_ControlSystem_Tx.py
    """
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    # Open socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((JETSON_IP, JETSON_PORT))
    print(f"Connected to {JETSON_IP}:{JETSON_PORT}")

    # MDS client & sim startup
    client = MdsClient(address="127.0.0.1", port=47474)
    client.load_scenario("6DoF_PE_Scenario.json")
    client.set_time_scale(1)
    logging.info("Stabilizing for 5 s…")
    time.sleep(5)

    # Record sim start-time
    _, sim_start_ms = client.get_current_time()
    logging.info("Sim clock start = %d ms", sim_start_ms)

    # Start CV feed
    CV_FEED_SIZE = 1024
    feed_cv = setup_video_feed(client, CV_FEED_SIZE)

    # Right here, grab the desired quaternion for all future frames
    _, first_rel = client.get_relative_to_camera_position("TANGO", feed_cv)
    q0 = first_rel.rotation
    desired_q = [q0.x, q0.y, q0.z, q0.w]

    # Launch control orchestrator
    docking_event = threading.Event()
    threading.Thread(target=control_orchestrator,
                     args=(client, docking_event),
                     daemon=True).start()
    
    # Create logger
    logger = ErrorLogger()

    # CV transmit loop + 2 min simulated-time watchdog
    period = 1.0 / FPS
    next_t = time.time()
    frame_ct = 0

    # Prepare a one‐time TR timestamp
    tr_start_ms = None

    try:
        while True:
            # Check simulated-time cutoff first
            _, now_ms = client.get_current_time()

            # If TR has started but we haven't yet recorded its t0, do it now
            if tr_started and tr_start_ms is None:
                tr_start_ms = now_ms
                logging.info("TR phase begun—starting 115 000 ms countdown at t=%d", tr_start_ms)

            # Once we have t0, enforce the 115_000 ms cutoff
            if tr_start_ms is not None and now_ms - tr_start_ms >= 110_000:
                logging.info("2 min after TR start → docking")
                docking_event.set()
                break

            # Throttle to FPS
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            # Grab JPEG frame + pose
            reply, _, img_jpeg = client.get_next_video_frame(feed_cv)
            if not img_jpeg:
                continue

            _, rel = client.get_relative_to_camera_position("TANGO", feed_cv)
            p, r = rel.position, rel.rotation
            sim_time = (now_ms - sim_start_ms) / 1000.0

            # Send timestamp + pose
            sock.sendall(struct.pack(
                '>d7f',
                sim_time,
                p.x, p.y, p.z,
                r.x, r.y, r.z, r.w
            ))

            # Encode PNG + send
            arr = np.frombuffer(img_jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            ok, buf = cv2.imencode('.png', img)
            if not ok:
                continue
            png = buf.tobytes()
            sock.sendall(struct.pack('>I', len(png)))
            sock.sendall(png)

            frame_ct += 1
            print(f"[{frame_ct:04d}] Sent frame @ {sim_time:.3f}s")
            log_frame_errors(sim_time, frame_ct, p, r, desired_q, logger)

    except KeyboardInterrupt:
        pass

    finally:
        logger.save(
            trans_filename='translation_errors.csv',
            att_filename='attitude_errors.csv'
        )

        # Teardown
        sock.close()
        client.stop_video_feed(feed_cv)
        client.close()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    main()
