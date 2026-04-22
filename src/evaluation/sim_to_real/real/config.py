"""Real-drone config for the RQ6 sim-to-real path."""

import os
import logging

from evaluation.sim_to_real.common.config import (
    ACT_DIM,
    ACTOR_FC_LAYERS,
    ACTOR_HIDDEN_DIM,
    ACTOR_LSTM_LAYERS,
    ACTOR_OBS_DIM,
    CRITIC_FC_LAYERS,
    CRITIC_HIDDEN_DIM,
    CRITIC_LSTM_LAYERS,
    CRITIC_OBS_DIM,
    FULL_ACT_DIM,
    FULL_OBS_DIM,
    HIL_STATE_DIM,
    LSTM_HIDDEN_DIM_ACTOR,
    LSTM_HIDDEN_DIM_CRITIC,
    MAX_BIAS,
    MAX_SEQ_LEN,
    METERS_PER_DEGREE_LAT,
    PRIVILEGED_INFO_DIM,
    STANDARD_GRAVITY,
    ZERO_ACTION,
)

NUM_EVAL_WORKERS = 1
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
LOG_DIR = os.environ.get(
    "GAP_RAY_LOG_DIR",
    os.path.join(PROJECT_ROOT, "results", "fresh", "ray_results"),
)
PX4_PARAM_SPEED = "1"

# Real-flight timing assumptions:
# - GPS/MAVLink observation path is assumed to be available at about 8 Hz.
# - One attack action is held for 1 s.
# - One episode times out after 300 s.
# - MAV_TIMEOUT tolerates up to ~8 missed 8 Hz samples.
MAV_HERTZ = 8
STEP_BASE = 1.0
EP_TIMEOUT = 300.0
MAV_TIMEOUT = 8 / MAV_HERTZ

# Site/mission parameters for the real test:
# - target location
# - current hover altitude / target altitude
# - horizontal target distance
TARGET_LAT = 36.01351
TARGET_LON = 129.31921
TAKEOFF_ALT = 10.0
TARGET_DISTANCE = 60.0

# Success/geofence parameters:
# - 10 m cylinder success region
# - horizontal and vertical safety fence
SUCCESS_RADIUS_M = 10.0
SAFE_TARGET_DIST = 10.0 + TARGET_DISTANCE
SAFE_ALT_MIN = 0.0
SAFE_ALT_MAX = 40.0

SUCCESS_RADIUS_BONUS = 20.0
TRUNCATED_PENALTY = -20.0
TIME_PENALTY = -0.01
DISTANCE_K = 0.1
VELOCITY_K = 0.025

LOG_LEVEL_PERFORMANCE = "WARNING"
logging.basicConfig(
    level=LOG_LEVEL_PERFORMANCE,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_TYPES = ["GPS_RAW_INT", "ATTITUDE_QUATERNION", "SCALED_IMU"]
