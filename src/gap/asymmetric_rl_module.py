import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.apis.target_network_api import (
    TargetNetworkAPI,
    TARGET_NETWORK_ACTION_DIST_INPUTS,
)
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

from .config import (
    ACTOR_HIDDEN_DIM,
    CRITIC_HIDDEN_DIM,
    ACTOR_FC_LAYERS,
    CRITIC_FC_LAYERS,
    LSTM_HIDDEN_DIM_ACTOR,
    LSTM_HIDDEN_DIM_CRITIC,
    ACTOR_LSTM_LAYERS,
    CRITIC_LSTM_LAYERS,
    logger,
)

import os

SHAPE_DEBUG = os.getenv("RL_DEBUG_SHAPES", "0") == "1"

def _maybe_flatten_lstm(*lstms):
    for l in lstms:
        try:
            if hasattr(l, "flatten_parameters"):
                l.flatten_parameters()
        except Exception:
            pass

def _assert_finite(name: str, *tensors: torch.Tensor):
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            raise FloatingPointError(f"{name}: found NaN/Inf in tensor with shape {tuple(t.shape)}")

def _shape(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return tuple(x.shape) + (str(x.device),)
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return tuple(x.shape)
    except Exception:
        pass
    return type(x).__name__

def _dump_batch_keys(tag, batch_dict):
    try:
        from ray.rllib.core.columns import Columns
        obs = batch_dict.get(Columns.OBS, {})
        sin = batch_dict.get(Columns.STATE_IN, {})
        out = batch_dict.get(Columns.STATE_OUT, {})
        parts = [
            f"{tag}::OBS keys={list(getattr(obs, 'keys', lambda: [])())}",
            f"{tag}::STATE_IN keys={list(getattr(sin, 'keys', lambda: [])())}",
            f"{tag}::STATE_OUT keys={list(getattr(out, 'keys', lambda: [])())}",
        ]
        logger.debug(" | ".join(parts))
    except Exception as e:
        logger.debug(f"{tag}::dump_keys failed: {e}")

def _make_mlp(in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers = [nn.LayerNorm(in_dim)]
    last = in_dim
    for _ in range(num_layers):
        layers.append(SlimFC(last, hidden_dim, activation_fn=nn.ReLU))
        last = hidden_dim
    return nn.Sequential(*layers)

def _to_BLH_for_return(x, num_layers):
    if x.dim() == 3 and x.shape[0] == num_layers:
        x = x.transpose(0, 1)
    elif x.dim() == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3, f"state_out must be (B,L,H), got {tuple(x.shape)}"
    return x.contiguous()

class AsymmetricLSTMModule(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """
    Asymmetric actor-critic RLlib module for GAP.

    The actor runs on external observations only. The critic sees the larger
    privileged state during training only. Most of the shape handling here is
    RLlib state plumbing for the LSTMs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(TorchRLModule)
    def setup(self) -> None:
        logger.info("[SETUP] AsymmetricLSTMModule: start")

        obs_space = self.observation_space
        action_space = self.action_space
        self.actor_obs_dim = obs_space["actor"].shape[0]
        self.critic_obs_dim = obs_space["critic"].shape[0]
        self.num_outputs = action_space.shape[0]

        self.actor_encoder = _make_mlp(
            in_dim=self.actor_obs_dim,
            hidden_dim=ACTOR_HIDDEN_DIM,
            num_layers=ACTOR_FC_LAYERS,
        )
        self.actor_lstm = nn.LSTM(
            input_size=ACTOR_HIDDEN_DIM,
            hidden_size=LSTM_HIDDEN_DIM_ACTOR,
            num_layers=ACTOR_LSTM_LAYERS,
            batch_first=False,
        )
        self.actor_logits = SlimFC(LSTM_HIDDEN_DIM_ACTOR, self.num_outputs * 2, activation_fn=None)

        self.critic_encoder = _make_mlp(
            in_dim=self.critic_obs_dim,
            hidden_dim=CRITIC_HIDDEN_DIM,
            num_layers=CRITIC_FC_LAYERS,
        )
        self.critic_lstm = nn.LSTM(
            input_size=CRITIC_HIDDEN_DIM,
            hidden_size=LSTM_HIDDEN_DIM_CRITIC,
            num_layers=CRITIC_LSTM_LAYERS,
            batch_first=False,
        )
        self.critic_vf = SlimFC(LSTM_HIDDEN_DIM_CRITIC, 1, activation_fn=None)

        self.action_dist_cls = TorchDiagGaussian
        self.actor_lstm.flatten_parameters()
        self.critic_lstm.flatten_parameters()

        logger.info(
            f"AsymmetricLSTMModule initialized | "
            f"ActorLSTM: L={ACTOR_LSTM_LAYERS}, H={LSTM_HIDDEN_DIM_ACTOR} | "
            f"CriticLSTM: L={CRITIC_LSTM_LAYERS}, H={LSTM_HIDDEN_DIM_CRITIC}"
        )

        logger.info("[SETUP] AsymmetricLSTMModule: done")

    def _initial_actor_state_no_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the no-batch state format RLlib expects."""
        if ACTOR_LSTM_LAYERS == 1:
            h = torch.zeros(LSTM_HIDDEN_DIM_ACTOR, device=device, dtype=torch.float32)
            c = torch.zeros(LSTM_HIDDEN_DIM_ACTOR, device=device, dtype=torch.float32)
        else:
            h = torch.zeros(ACTOR_LSTM_LAYERS, LSTM_HIDDEN_DIM_ACTOR, device=device, dtype=torch.float32)
            c = torch.zeros(ACTOR_LSTM_LAYERS, LSTM_HIDDEN_DIM_ACTOR, device=device, dtype=torch.float32)
        return h, c

    def _actor_state_no_batch_to_lbh(
        self, h: torch.Tensor, c: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert RLlib's no-batch state into (L, B, H)."""
        L = ACTOR_LSTM_LAYERS
        H = LSTM_HIDDEN_DIM_ACTOR
        if h.dim() == 1:
            h = h.view(1, 1, H).expand(L, batch_size, H)
            c = c.view(1, 1, H).expand(L, batch_size, H)
        elif h.dim() == 2:
            if h.shape[0] != L or h.shape[1] != H:
                raise ValueError(f"Actor state shape mismatch: expect (L,H)=({L},{H}) got {tuple(h.shape)}")
            h = h.unsqueeze(1).expand(L, batch_size, H)
            c = c.unsqueeze(1).expand(L, batch_size, H)
        else:
            raise ValueError(f"Actor state must be (H,) or (L,H), got {tuple(h.shape)}")
        return h.contiguous(), c.contiguous()

    def _actor_state_lbh_to_no_batch(
        self, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert (L, B, H) back into RLlib's no-batch state format."""
        L = ACTOR_LSTM_LAYERS
        H = LSTM_HIDDEN_DIM_ACTOR
        if h.dim() != 3 or c.dim() != 3:
            raise ValueError(f"LSTM state must be 3D (L,B,H), got {tuple(h.shape)}, {tuple(c.shape)}")
        if h.shape[0] != L or h.shape[2] != H:
            raise ValueError(f"LSTM state shape mismatch: got {tuple(h.shape)} expect (L,B,H) with L={L},H={H}")
        B = h.shape[1]
        if B != 1:
            raise RuntimeError(f"Expected batch size B=1 in inference; got B={B}")
        h = h[:, 0, :]
        c = c[:, 0, :]
        if L == 1:
            h = h.squeeze(0)
            c = c.squeeze(0)
        return h.contiguous(), c.contiguous()

    def _time_distributed(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply an MLP independently to each timestep."""
        B, T, D = x.shape
        y = module(x.reshape(B * T, D))
        H = y.shape[-1]
        return y.view(B, T, H)

    def _parse_actor_state_in(self, h_in: torch.Tensor, c_in: torch.Tensor, B: int, dev: torch.device):
        """Normalize RLlib actor state into (L, B, H)."""
        L = ACTOR_LSTM_LAYERS
        H = LSTM_HIDDEN_DIM_ACTOR

        if h_in is None or c_in is None:
            h = torch.zeros(L, B, H, device=dev)
            c = torch.zeros(L, B, H, device=dev)
            return h, c, "ZERO"

        h = torch.as_tensor(h_in, device=dev, dtype=torch.float32)
        c = torch.as_tensor(c_in, device=dev, dtype=torch.float32)

        if h.dim() == 4 and h.shape[0] == 1 and h.shape[2] == L and h.shape[3] == H:
            h = h.squeeze(0).permute(1, 0, 2).contiguous()
            c = c.squeeze(0).permute(1, 0, 2).contiguous()
            return h, c, "T_B_L_H"

        if h.dim() == 1 and h.shape[0] == H and L == 1:
            h = h.view(1, 1, H).expand(1, B, H)
            c = c.view(1, 1, H).expand(1, B, H)
            return h.contiguous(), c.contiguous(), "H"

        if h.dim() == 2 and h.shape == (L, H):
            h = h.unsqueeze(1).expand(L, B, H)
            c = c.unsqueeze(1).expand(L, B, H)
            return h.contiguous(), c.contiguous(), "L_H"

        if h.dim() == 3 and h.shape == (B, L, H):
            h = h.permute(1, 0, 2).contiguous()
            c = c.permute(1, 0, 2).contiguous()
            return h, c, "B_L_H"

        if h.dim() == 3 and h.shape == (L, B, H):
            return h.contiguous(), c.contiguous(), "L_B_H"

        if h.dim() == 3 and h.shape[0] == 1 and h.shape[1] == L and h.shape[2] == H and B == 1:
            h = h.permute(1, 0, 2).contiguous()
            c = c.permute(1, 0, 2).contiguous()
            return h, c, "B_L_H"

        raise ValueError(
            f"Unsupported STATE_IN shape for actor_h: {tuple(h.shape)}; "
            f"expected (H,), (L,H), (B,L,H), (L,B,H), or (1,B,L,H)."
        )

    def _format_actor_state_out(self, h_lbh: torch.Tensor, c_lbh: torch.Tensor, B: int):
        h_blh = h_lbh.transpose(0, 1).contiguous()
        c_blh = c_lbh.transpose(0, 1).contiguous()
        return h_blh, c_blh

    @override(TorchRLModule)
    def get_initial_state(self) -> dict:
        h, c = self._initial_actor_state_no_batch(device=torch.device("cpu"))
        return {"actor_h": h, "actor_c": c}

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            dev = next(self.parameters()).device
            obs_actor = batch[Columns.OBS]["actor"].to(dev)
            if obs_actor.dim() == 3:
                obs_actor = obs_actor[:, -1, :]
            elif obs_actor.dim() != 2:
                raise ValueError(f"obs_actor must be 2D/3D, got {tuple(obs_actor.shape)}")
            B = obs_actor.shape[0]

            if SHAPE_DEBUG:
                _dump_batch_keys("INF-BATCH", batch)
                logger.debug(f"INF-IN-OBS:: obs_actor:{_shape(obs_actor)}")

            sin = batch.get(Columns.STATE_IN, {})
            h0, c0, _src = self._parse_actor_state_in(
                sin.get("actor_h"), sin.get("actor_c"), B, dev
            )

            if SHAPE_DEBUG:
                logger.debug(
                    f"INF-STATE-IN:: h:{_shape(sin.get('actor_h'))} "
                    f"c:{_shape(sin.get('actor_c'))} src={_src}"
                )
                logger.debug(f"INF-STATE0-LBH:: h0:{_shape(h0)} | c0:{_shape(c0)}")

            x = self.actor_encoder(obs_actor)
            x = x.unsqueeze(0).contiguous()
            _maybe_flatten_lstm(self.actor_lstm)
            x, (h_out, c_out) = self.actor_lstm(x, (h0, c0))

            x_last = x.squeeze(0)
            logits = self.actor_logits(x_last)
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            log_std = torch.clamp(log_std, min=-5.0, max=2.0)
            logits = torch.cat([mean, log_std], dim=-1)
            logits = logits.unsqueeze(0).contiguous()

            h_ret = _to_BLH_for_return(h_out, ACTOR_LSTM_LAYERS)
            c_ret = _to_BLH_for_return(c_out, ACTOR_LSTM_LAYERS)

            _assert_finite("INF-LOGITS", logits)
            assert logits.dim() == 3 and logits.shape[0] == 1, \
                f"expect logits (1,B,2A); got {tuple(logits.shape)}"
            assert h_ret.dim() == 3 and h_ret.shape[1] == ACTOR_LSTM_LAYERS, \
                f"expect state_out (B,L,H); got {tuple(h_ret.shape)}, {tuple(c_ret.shape)}"

            if SHAPE_DEBUG and getattr(self, "_dbg_cnt", 0) < 10:
                logger.debug(f"INF-RET:: logits:{_shape(logits)} | actor_h:{_shape(h_ret)} | actor_c:{_shape(c_ret)}")
                self._dbg_cnt = getattr(self, "_dbg_cnt", 0) + 1


            return {
                Columns.ACTION_DIST_INPUTS: logits,
                Columns.STATE_OUT: {
                    "actor_h": h_ret,
                    "actor_c": c_ret,
                },
            }
        except Exception as e:
            import traceback
            logger.error("[INF-EXC] %s: %s", type(e).__name__, str(e))
            logger.debug(traceback.format_exc())
            raise

    @override(RLModule)
    def _forward_train(self, batch: dict, **kwargs) -> dict:
        try:
            dev = next(self.parameters()).device
            obs_actor  = batch[Columns.OBS]["actor"].to(dev)
            obs_critic = batch[Columns.OBS]["critic"].to(dev)

            if SHAPE_DEBUG:
                _dump_batch_keys("TRN-BATCH", batch)
                logger.debug(f"TRN-IN:: obs_actor:{_shape(obs_actor)} | obs_critic:{_shape(obs_critic)}")

            squeeze_time = False
            if obs_actor.dim() == 2:
                obs_actor  = obs_actor.unsqueeze(1)
                obs_critic = obs_critic.unsqueeze(1)
                squeeze_time = True
            elif obs_actor.dim() != 3:
                raise ValueError(f"obs_actor expected 2D/3D, got {tuple(obs_actor.shape)}")

            B, T, _ = obs_actor.shape

            xa = self._time_distributed(self.actor_encoder, obs_actor)
            xa = xa.permute(1, 0, 2).contiguous()
            h0_a = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            c0_a = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            _maybe_flatten_lstm(self.actor_lstm)
            xa, _ = self.actor_lstm(xa, (h0_a, c0_a))
            xa = xa.permute(1, 0, 2).contiguous()
            logits = self.actor_logits(xa)
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            log_std = torch.clamp(log_std, min=-5.0, max=2.0)
            logits = torch.cat([mean, log_std], dim=-1)

            xc = self._time_distributed(self.critic_encoder, obs_critic)
            xc = xc.permute(1, 0, 2).contiguous()
            h0_c = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            c0_c = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            _maybe_flatten_lstm(self.critic_lstm)
            xc, _ = self.critic_lstm(xc, (h0_c, c0_c))
            xc = xc.permute(1, 0, 2).contiguous()
            values = self.critic_vf(xc).squeeze(-1)

            if squeeze_time:
                logits = logits.squeeze(1)
                values = values.squeeze(1)

            if SHAPE_DEBUG:
                logger.debug(f"TRN-OUT:: logits:{_shape(logits)} | values:{_shape(values)}")

            _assert_finite("TRN-LOGITS/VALUES", logits, values)

            return {Columns.ACTION_DIST_INPUTS: logits, Columns.VF_PREDS: values}
        except Exception as e:
            import traceback
            logger.error("[TRN-EXC] %s: %s", type(e).__name__, str(e))
            logger.debug(traceback.format_exc())
            raise

    @override(RLModule)
    def _forward_exploration(self, batch: dict, **kwargs) -> dict:
        return self._forward_inference(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings=None):
        dev = next(self.parameters()).device
        obs_critic = batch[Columns.OBS]["critic"].to(dev)

        if obs_critic.dim() == 2:
            B = obs_critic.shape[0]
            x = self.critic_encoder(obs_critic)
            x = x.unsqueeze(0)
            h0 = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            c0 = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            _maybe_flatten_lstm(self.critic_lstm)
            x, _ = self.critic_lstm(x, (h0, c0))
            v = self.critic_vf(x.squeeze(0)).squeeze(-1)
            return v

        if obs_critic.dim() == 3:
            B, T, _ = obs_critic.shape
            x = self._time_distributed(self.critic_encoder, obs_critic)
            x = x.permute(1, 0, 2).contiguous()
            h0 = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            c0 = torch.zeros(CRITIC_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_CRITIC, device=dev)
            _maybe_flatten_lstm(self.critic_lstm)
            x, _ = self.critic_lstm(x, (h0, c0))
            x = x.permute(1, 0, 2).contiguous()
            v = self.critic_vf(x).squeeze(-1)
            return v

        raise ValueError(f"obs_critic expected 2D/3D, got {tuple(obs_critic.shape)}")
    @override(TargetNetworkAPI)
    def make_target_networks(self):
        self.target_actor_encoder = make_target_network(self.actor_encoder)
        self.target_actor_lstm = make_target_network(self.actor_lstm)
        self.target_actor_logits = make_target_network(self.actor_logits)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self):
        return [
            (self.actor_encoder, self.target_actor_encoder),
            (self.actor_lstm, self.target_actor_lstm),
            (self.actor_logits, self.target_actor_logits),
        ]

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any]):
        dev = next(self.parameters()).device
        obs_actor = batch[Columns.OBS]["actor"].to(dev)

        if obs_actor.dim() == 2:
            B = obs_actor.shape[0]
            x = self.target_actor_encoder(obs_actor)
            x = x.unsqueeze(0)
            h0 = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            c0 = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            _maybe_flatten_lstm(self.target_actor_lstm)
            x, _ = self.target_actor_lstm(x, (h0, c0))
            logits = self.target_actor_logits(x.squeeze(0))
        elif obs_actor.dim() == 3:
            B, T, _ = obs_actor.shape
            x = self.target_actor_encoder(obs_actor.reshape(B * T, obs_actor.shape[-1]))
            x = x.view(B, T, -1).permute(1, 0, 2).contiguous()
            h0 = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            c0 = torch.zeros(ACTOR_LSTM_LAYERS, B, LSTM_HIDDEN_DIM_ACTOR, device=dev)
            _maybe_flatten_lstm(self.target_actor_lstm)
            x, _ = self.target_actor_lstm(x, (h0, c0))
            x = x.permute(1, 0, 2).contiguous()
            logits = self.target_actor_logits(x)
        else:
            raise ValueError(f"obs_actor must be 2D or 3D, got {obs_actor.dim()}D")

        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        logits = torch.cat([mean, log_std], dim=-1)

        _assert_finite("TARGET-LOGITS", logits)

        return {TARGET_NETWORK_ACTION_DIST_INPUTS: logits}
