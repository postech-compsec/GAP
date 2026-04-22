"""Sim-only config for the RQ6 sim-to-real SITL evaluator."""

import logging
import os

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

NUM_WORKERS = 30
NUM_EVAL_WORKERS = 12
GPU_WORKER = 0.1

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

LOG_DIR = os.environ.get(
    "GAP_RAY_LOG_DIR",
    os.path.join(_PROJECT_ROOT, "results", "fresh", "ray_results"),
)

PX4_PARAM_HEADLESS = "1"
PX4_PARAM_SPEED = "4"
MAV_HERTZ = 8

STEP_BASE = 1.0 / int(PX4_PARAM_SPEED)
EP_TIMEOUT = 300.0 / int(PX4_PARAM_SPEED)
MAV_TIMEOUT = 3 / (MAV_HERTZ * int(PX4_PARAM_SPEED))

TARGET_LAT = float(os.environ.get("SIM_TO_REAL_TARGET_LAT", "47.39855040647849"))
TARGET_LON = float(os.environ.get("SIM_TO_REAL_TARGET_LON", "8.545290332727657"))
TAKEOFF_ALT = 60.0
SUCCESS_RADIUS_M = 10.0
RANDOM_START_RADIUS_M = 220.0
SAFE_TARGET_DIST = 50.0 + RANDOM_START_RADIUS_M
SAFE_ALT_MIN = 0.0
SAFE_ALT_MAX = 120.0

SUCCESS_RADIUS_BONUS = 20.0
TRUNCATED_PENALTY = -20.0
TIME_PENALTY = -0.01
DISTANCE_K = 0.1
VELOCITY_K = 0.025

PX4_ROOT = os.environ.get("PX4_ROOT", os.path.join(_PROJECT_ROOT, "GAP-PX4-Autopilot"))
PX4_RUN_SCRIPT = os.path.join(os.path.dirname(__file__), "run_px4_multi.sh")
JMAVSIM_SCRIPT = os.path.join(PX4_ROOT, "Tools/simulation/jmavsim/jmavsim_run.sh")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
