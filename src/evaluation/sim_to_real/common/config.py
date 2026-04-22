"""Shared model constants for the sim-to-real pipeline."""

import logging
import numpy as np

# Must match `sim-to-real_model`.
HIL_STATE_DIM = 25
PRIVILEGED_INFO_DIM = 79
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

MAX_BIAS = 0.06

STANDARD_GRAVITY = 9.80665
METERS_PER_DEGREE_LAT = 111319.5

ZERO_ACTION = np.zeros(FULL_ACT_DIM, np.float32)

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
