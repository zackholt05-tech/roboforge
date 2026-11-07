"""Deterministic behavior-cloning networks: BC-MLP and BC-RNN (MASTER_SPEC §5.2-5.3).

Both take a normalized observation window and return a normalized action chunk:

* input ``obs_history [B, T_o, D]`` (oldest first) and ``obs_valid [B, T_o]``. Pre-start
  padding repeats the first observation with ``obs_valid = False`` (DECISIONS D-022); the
  newest step is always valid;
* output ``[B, T_p, A]`` in the normalized action space. Decoding to simulator actions
  (unnormalize, then clip to the ActionSpec bounds) happens outside the network, in
  :class:`roboforge.data.normalization.ActionCodec`, identically for every family.

**BC-MLP** flattens the window. It consumes padded steps exactly as the training windows
present them, so no mask is needed.

**BC-RNN** runs an LSTM (or GRU) **from a zero state over the valid steps of the window
only** and predicts from the hidden state of the newest step. Padded steps are a prefix, so
skipping them equals starting the recurrence at the episode start. At inference the policy
re-runs the recurrence over the last ``L`` observations at every decision, which is exactly
what each training window does. No hidden state persists between decisions or episodes.
This differs from robomimic's BC-RNN rollout, which carries the hidden state for up to
``L`` steps and then resets it (RESEARCH_PROTOCOL §6.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "mish": nn.Mish}
CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


@dataclass(frozen=True)
class MLPConfig:
    """BC-MLP architecture."""

    hidden: tuple[int, ...] = (1024, 1024)
    activation: str = "relu"
    layer_norm: bool = False
    dropout: float = 0.0


@dataclass(frozen=True)
class RNNConfig:
    """BC-RNN architecture (recurrent encoder of the observation window + linear head)."""

    cell: str = "lstm"
    hidden: int = 296
    layers: int = 2
    dropout: float = 0.0


def _check_window(obs_history: torch.Tensor, obs_valid: torch.Tensor, horizon: int,
                  obs_dim: int) -> None:
    if obs_history.ndim != 3 or obs_history.shape[1:] != (horizon, obs_dim):
        raise ValueError(f"obs_history shape {tuple(obs_history.shape)} != [B, {horizon}, "
                         f"{obs_dim}]")
    if obs_valid.shape != obs_history.shape[:2] or obs_valid.dtype != torch.bool:
        raise ValueError("obs_valid must be a bool tensor of shape [B, T_o]")


class BCMLP(nn.Module):
    """MLP from the flattened observation window to a normalized action chunk."""

    def __init__(self, obs_dim: int, action_dim: int, obs_horizon: int, pred_horizon: int,
                 cfg: MLPConfig) -> None:
        super().__init__()
        if cfg.activation not in ACTIVATIONS:
            raise ValueError(f"activation {cfg.activation!r} not in {sorted(ACTIVATIONS)}")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_horizon, self.pred_horizon, self.cfg = obs_horizon, pred_horizon, cfg
        layers: list[nn.Module] = []
        d = obs_dim * obs_horizon
        for h in cfg.hidden:
            layers.append(nn.Linear(d, h))
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(ACTIVATIONS[cfg.activation]())
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            d = h
        layers.append(nn.Linear(d, pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        """``[B, T_o, D]`` window (+ unused mask) -> ``[B, T_p, A]`` normalized actions."""
        _check_window(obs_history, obs_valid, self.obs_horizon, self.obs_dim)
        b = obs_history.shape[0]
        return self.net(obs_history.reshape(b, -1)).view(b, self.pred_horizon, self.action_dim)


class BCRNN(nn.Module):
    """Recurrent encoder over the valid steps of the window, linear head on the newest step."""

    def __init__(self, obs_dim: int, action_dim: int, obs_horizon: int, pred_horizon: int,
                 cfg: RNNConfig) -> None:
        super().__init__()
        if cfg.cell not in CELLS:
            raise ValueError(f"cell {cfg.cell!r} not in {sorted(CELLS)}")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_horizon, self.pred_horizon, self.cfg = obs_horizon, pred_horizon, cfg
        self.rnn = CELLS[cfg.cell](obs_dim, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                                   dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Linear(cfg.hidden, pred_horizon * action_dim)

    def encode(self, obs_history: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        """Top-layer hidden state after the newest step, starting from zero at the first
        valid step. Returns ``[B, hidden]``."""
        _check_window(obs_history, obs_valid, self.obs_horizon, self.obs_dim)
        if not bool(obs_valid[:, -1].all()):
            raise ValueError("the newest step of every window must be valid")
        if bool((obs_valid[:, 1:] < obs_valid[:, :-1]).any()):
            raise ValueError("obs_valid must be a padded prefix (False ... False, True ... True)")
        b, horizon = obs_valid.shape
        n_valid = obs_valid.sum(dim=1)
        # Move the valid suffix to the front, then pack: each sequence runs over its own
        # valid steps only, from a zero state.
        shift = (horizon - n_valid).unsqueeze(1)
        idx = (torch.arange(horizon, device=obs_valid.device).unsqueeze(0) + shift) % horizon
        aligned = torch.gather(obs_history, 1, idx.unsqueeze(-1).expand(-1, -1, self.obs_dim))
        packed = nn.utils.rnn.pack_padded_sequence(aligned, n_valid.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, state = self.rnn(packed)
        h_n = state[0] if isinstance(state, tuple) else state
        return h_n[-1]

    def forward(self, obs_history: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        """``[B, L, D]`` window + mask -> ``[B, T_p, A]`` normalized actions."""
        b = obs_history.shape[0]
        return self.head(self.encode(obs_history, obs_valid)).view(b, self.pred_horizon,
                                                                    self.action_dim)


FAMILIES: dict[str, tuple[type[nn.Module], type]] = {"bc_mlp": (BCMLP, MLPConfig),
                                                     "bc_rnn": (BCRNN, RNNConfig)}


def model_config(family: str, raw: dict[str, Any]) -> MLPConfig | RNNConfig:
    """Typed architecture config for ``family``; unknown or missing keys raise."""
    if family not in FAMILIES:
        raise ValueError(f"family {family!r} not in {sorted(FAMILIES)}")
    cls = FAMILIES[family][1]
    fields = set(cls.__dataclass_fields__)
    if set(raw) != fields:
        raise ValueError(f"{family} model keys {sorted(raw)} != {sorted(fields)}")
    vals = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**vals)


def build_model(family: str, model_cfg: dict[str, Any], obs_dim: int, action_dim: int,
                obs_horizon: int, pred_horizon: int) -> nn.Module:
    """Construct a family's network from its JSON-compatible config."""
    net_cls = FAMILIES[family][0]
    return net_cls(obs_dim, action_dim, obs_horizon, pred_horizon,
                   model_config(family, model_cfg))


def config_dict(cfg: MLPConfig | RNNConfig) -> dict[str, Any]:
    """JSON-compatible architecture config (tuples become lists)."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}


def count_parameters(model: nn.Module) -> int:
    """Exact number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
