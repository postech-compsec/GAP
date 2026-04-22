"""Shared GAP model constants."""

import logging

ACTOR_HIDDEN_DIM = 512
ACTOR_FC_LAYERS = 3
LSTM_HIDDEN_DIM_ACTOR = 512
ACTOR_LSTM_LAYERS = 2

CRITIC_HIDDEN_DIM = 1024
CRITIC_FC_LAYERS = 4
LSTM_HIDDEN_DIM_CRITIC = 1024
CRITIC_LSTM_LAYERS = 2

MAX_SEQ_LEN = 32

ACT_DIM = 2
FULL_ACT_DIM = 3

MAX_GYRO_BIAS = 0.06

logging.basicConfig(
    level="WARNING",
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("gap")
