"""Shared config for baseline, cross-platform, CI-detector, and RL tests."""

import os
import logging
import numpy as np


# Target / mission
TARGET_LAT = 47.39855040647849
TARGET_LON = 8.545290332727657
TAKEOFF_ALT = 60.0  # m relative

# In RL tests the drone sits at the target center and targets rotate around it.
DRONE_START_LAT = TARGET_LAT
DRONE_START_LON = TARGET_LON

SUCCESS_CRITERIA = {
    "sphere_20m": {"type": "sphere", "radius": 20.0},
    "cylinder_20m": {"type": "cylinder", "radius": 20.0},
    "sphere_10m": {"type": "sphere", "radius": 10.0},
    "cylinder_10m": {"type": "cylinder", "radius": 10.0},
}
# Paper's primary target (matches primary/config.py).
SUCCESS_RADIUS_M = SUCCESS_CRITERIA["cylinder_10m"]["radius"]

# Episode structure
RANDOM_START_RADIUS_M = 220.0
NUM_START_POSITIONS = 12
NUM_TRIALS_PER_POSITION = 2
TOTAL_TESTS = NUM_START_POSITIONS * NUM_TRIALS_PER_POSITION

# Simulation
SIMULATION_SPEEDUP = 4          # jMAVSim + ArduPilot support speedup; Gazebo does not.
TEST_TIMEOUT_SIM_SEC = 300
TEST_TIMEOUT_WALL_SEC = TEST_TIMEOUT_SIM_SEC / SIMULATION_SPEEDUP

# Attack
MAX_GYRO_BIAS = 0.06            # rad/s
ATTACK_RANGE_MIN = -MAX_GYRO_BIAS
ATTACK_RANGE_MAX = MAX_GYRO_BIAS
ATTACK_INTERVAL = 1.0           # s, baseline Case1

# Constants
STANDARD_GRAVITY = 9.80665
METERS_PER_DEGREE_LAT = 111319.5

# Paths — all env-overridable for non-default layouts.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
LOG_DIR = os.environ.get("GAP_LOG_DIR", os.path.join(PROJECT_ROOT, "results", "fresh", "log"))
PX4_ROOT = os.environ.get("PX4_ROOT", os.path.join(PROJECT_ROOT, "GAP-PX4-Autopilot"))
ARDUPILOT_ROOT = os.environ.get("ARDUPILOT_ROOT", os.path.join(PROJECT_ROOT, "GAP-ardupilot"))

# RL checkpoint
MODELS_DIR = os.environ.get(
    "GAP_MODELS_DIR",
    os.path.join(PROJECT_ROOT, "src", "gap", "models"),
)
GAP_MODEL_PATH = os.environ.get("GAP_MODEL_PATH", os.path.join(MODELS_DIR, "gap_model"))

# ArduPilot frames tested by RQ3.
ARDUPILOT_FRAMES = [
    "quad", "hexa", "octa", "octaquad", "y6",
    "dodeca-hexa", "tri", "singlecopter", "coaxcopter",
]

# Logging
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
