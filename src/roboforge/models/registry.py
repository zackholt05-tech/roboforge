"""Policy-family registry: network class, typed architecture config and loss kind.

Families:

* ``bc_mlp``, ``bc_rnn``: deterministic BC (:mod:`roboforge.models.bc`), loss ``mse``;
* ``diffusion_unet``: conditional diffusion (:mod:`roboforge.models.diffusion`), loss
  ``denoising_mse``. Its networks are ``stochastic``: they sample with an explicit generator.

Configs, trainers and checkpoints resolve families only through this module, so the BC
configs and their hashes are unchanged by the addition of diffusion.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from torch import nn

from roboforge.models.bc import BCMLP, BCRNN, MLPConfig, RNNConfig, count_parameters
from roboforge.models.diffusion import DiffusionConfig, DiffusionPolicy

FAMILIES: dict[str, tuple[type[nn.Module], type]] = {
    "bc_mlp": (BCMLP, MLPConfig), "bc_rnn": (BCRNN, RNNConfig),
    "diffusion_unet": (DiffusionPolicy, DiffusionConfig)}
LOSS_KINDS = {"bc_mlp": "mse", "bc_rnn": "mse", "diffusion_unet": "denoising_mse"}
ModelConfig = MLPConfig | RNNConfig | DiffusionConfig


def model_config(family: str, raw: dict[str, Any]) -> ModelConfig:
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


def config_dict(cfg: ModelConfig) -> dict[str, Any]:
    """JSON-compatible architecture config (tuples become lists)."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}


def is_stochastic(model: nn.Module) -> bool:
    """True for networks that sample with an explicit generator (diffusion)."""
    return bool(getattr(model, "stochastic", False))


__all__ = ["FAMILIES", "LOSS_KINDS", "build_model", "config_dict", "count_parameters",
           "is_stochastic", "model_config"]
