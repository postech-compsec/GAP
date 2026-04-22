import os
import logging
import numpy as np

NUM_WORKERS = 30
NUM_EVAL_WORKERS = 12
GPU_WORKER = 0.1

# Override with GAP_RAY_LOG_DIR if needed.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.environ.get(
    "GAP_RAY_LOG_DIR",
    os.path.join(_PROJECT_ROOT, "results", "fresh", "ray_results"),
)

HIL_STATE_DIM = 21
PRIVILEGED_INFO_DIM = 55

ACT_DIM = 2
FULL_ACT_DIM = 3

FULL_OBS_DIM = HIL_STATE_DIM + PRIVILEGED_INFO_DIM 
ACTOR_OBS_DIM = HIL_STATE_DIM + FULL_ACT_DIM
CRITIC_OBS_DIM = FULL_OBS_DIM + FULL_ACT_DIM

ACTOR_HIDDEN_DIM = 512
ACTOR_FC_LAYERS = 3
LSTM_HIDDEN_DIM_ACTOR = 512
ACTOR_LSTM_LAYERS = 2

CRITIC_HIDDEN_DIM = 1024
CRITIC_FC_LAYERS = 4
LSTM_HIDDEN_DIM_CRITIC = 1024
CRITIC_LSTM_LAYERS = 2

MAX_SEQ_LEN = 32

PX4_PARAM_HEADLESS = "1"
PX4_PARAM_SPEED = "4"
MAV_HERTZ = 20

STEP_BASE = 1.0 / int(PX4_PARAM_SPEED)
EP_TIMEOUT = 300.0 / int(PX4_PARAM_SPEED)
MAV_TIMEOUT = 4 / (MAV_HERTZ * int(PX4_PARAM_SPEED))

TARGET_LAT = 47.39855040647849
TARGET_LON = 8.545290332727657
TAKEOFF_ALT = 60.0

SUCCESS_RADIUS_M = 10.0
RANDOM_START_RADIUS_M = 220.0

SAFE_TARGET_DIST = 50.0 + RANDOM_START_RADIUS_M
SAFE_ALT_MIN = 0.0
SAFE_ALT_MAX = 120.0

MAX_BIAS = 0.06

SUCCESS_RADIUS_BONUS = 20.0
TRUNCATED_PENALTY = -20.0
TIME_PENALTY = -0.01
DISTANCE_K = 0.1
VELOCITY_K = 0.025

STANDARD_GRAVITY = 9.80665
METERS_PER_DEGREE_LAT = 111319.5

ZERO_ACTION = np.zeros(FULL_ACT_DIM, np.float32)

# Override with PX4_ROOT to use an external checkout.
PX4_ROOT = os.environ.get("PX4_ROOT", os.path.join(_PROJECT_ROOT, "GAP-PX4-Autopilot"))
PX4_RUN_SCRIPT = os.path.join(os.path.dirname(__file__), 'run_px4_multi.sh')
JMAVSIM_SCRIPT = os.path.join(PX4_ROOT, 'Tools/simulation/jmavsim/jmavsim_run.sh')

LOG_LEVEL_PERFORMANCE = "WARNING"

logging.basicConfig(
    level=LOG_LEVEL_PERFORMANCE,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
