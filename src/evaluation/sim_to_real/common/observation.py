"""Shared observation processing for the sim-to-real pipeline.

The 25-dim processed_hil layout matches the sim-to-real model's training-time
observation. Both sim/gym_env.py (SITL) and real/gym_env.py (physical drone)
must emit the exact same vector.

Inputs:
    hil_state (ndarray[16]): [lat, lon, alt, vx, vy, vz, ax, ay, az,
                              qw, qx, qy, qz, wx, wy, wz] — same layout
        PX4's HIL_STATE_QUATERNION uses. On the physical drone the same
        layout is assembled from GPS_RAW_INT + ATTITUDE_QUATERNION +
        SCALED_IMU; in SITL it comes from
        GPS_RAW_INT + ATTITUDE_QUATERNION + SCALED_IMU.
    initial_hil (ndarray[16]): the hil_state captured pre-takeoff (used as
        the altitude reference for the relative-z calculation).
    episode_start_time (float): time.monotonic() timestamp at episode start.
"""

from __future__ import annotations

import time
import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation


def normalize_angle(angle):
    """Radians → [-1, +1]."""
    return ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi


def process_observation(
    hil_state,
    initial_hil,
    episode_start_time,
    *,
    target_lat,
    target_lon,
    takeoff_alt,
    ep_timeout,
    success_radius_m,
    safe_alt_min,
    safe_alt_max,
    safe_target_dist,
):
    """Return the 25-dim processed HIL used by the sim-to-real model."""
    lat, lon, alt_amsl = hil_state[0:3]
    vx, vy, vz = hil_state[3:6]
    ax, ay, az = hil_state[6:9]
    quat_frd_to_ned = hil_state[9:13]
    roll_speed, pitch_speed, yaw_speed = hil_state[13:16]

    time_since_reset = time.monotonic() - episode_start_time
    normalized_time = time_since_reset / ep_timeout

    geodesic = Geodesic.WGS84.Inverse(lat, lon, target_lat, target_lon)
    direction_to_target_ned = np.deg2rad(geodesic["azi1"])
    distance_to_target = geodesic["s12"]  # 2D horizontal
    north = distance_to_target * np.cos(direction_to_target_ned)
    east = distance_to_target * np.sin(direction_to_target_ned)
    relative_altitude = alt_amsl - initial_hil[2]
    z_to_target_downframe = relative_altitude - takeoff_alt
    rel_pos_ned = np.array([north, east, z_to_target_downframe], np.float32)
    vel_ned = np.array([vx, vy, vz], np.float32)

    if np.allclose(quat_frd_to_ned, 0.0):
        quat_frd_to_ned = np.array([1.0, 0.0, 0.0, 0.0])
    rot = Rotation.from_quat(quat_frd_to_ned, scalar_first=True)
    rel_pos_frd, vel_frd = rot.apply(
        np.array([rel_pos_ned, vel_ned]), inverse=True,
    )

    roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
    (
        n_roll, n_pitch, n_yaw, heading_error, n_direction_to_target,
    ) = normalize_angle(np.array([
        roll, pitch, yaw, direction_to_target_ned - yaw, direction_to_target_ned,
    ]))

    distance_3d = float(np.sqrt(distance_to_target ** 2 + z_to_target_downframe ** 2))
    if distance_3d > 1e-6:
        unit = rel_pos_ned[:3] / distance_3d
        closing_velocity = float(np.dot(vel_ned[:3], unit))
    else:
        closing_velocity = 0.0

    distance_to_alt_limit = (relative_altitude - safe_alt_min) / (safe_alt_max - safe_alt_min)
    distance_to_geo_fence = distance_to_target / safe_target_dist

    is_in_sphere = 1 if distance_to_target <= success_radius_m else 0
    total_velocity = float(np.linalg.norm(vel_frd))

    return np.array([
        normalized_time,                       # 0
        *rel_pos_frd,                          # 1-3
        *vel_frd,                              # 4-6
        ax, ay, az,                            # 7-9
        n_roll, n_pitch, n_yaw,                # 10-12
        roll_speed, pitch_speed, yaw_speed,    # 13-15
        distance_to_target,                    # 16 — 2D horizontal (success criterion)
        distance_3d,                           # 17
        closing_velocity,                      # 18
        n_direction_to_target,                 # 19
        heading_error,                         # 20
        distance_to_alt_limit,                 # 21
        distance_to_geo_fence,                 # 22
        is_in_sphere,                          # 23
        total_velocity,                        # 24
    ], np.float32)


def build_dict_observation(processed_hil, processed_priv, action,
                           actor_obs_dim, critic_obs_dim):
    """Assemble {actor, critic} dict obs matching the model's trained spaces."""
    actor_obs = np.concatenate([processed_hil, action]).astype(np.float32)
    critic_obs = np.concatenate(
        [processed_hil, processed_priv, action]
    ).astype(np.float32)
    return {"actor": actor_obs, "critic": critic_obs}
