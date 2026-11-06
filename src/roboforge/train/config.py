"""Typed, strictly validated training-run configuration (RESEARCH_PROTOCOL §6).

A run config is a YAML file (``configs/models/<family>.yaml``) plus optional
``--set section.key=value`` overrides. Unknown or missing keys raise, so a typo can never
silently fall back to a default. The fully resolved config is stored in every checkpoint and
run manifest, and its canonical-JSON SHA-256 is part of the run ID.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from roboforge.models.bc import FAMILIES, config_dict, model_config
from roboforge.provenance import sha256_json

SCHEMA = "roboforge.train_config/1"
SECTIONS = ("horizons", "model", "loss", "optim", "train")
OPTIMIZERS = ("adamw",)
SCHEDULES = ("constant", "cosine")
LOSSES = ("mse",)


@dataclass(frozen=True)
class Horizons:
    """Observation window ``obs`` (T_o, or the BC-RNN context L), prediction ``pred`` (T_p)
    and execution ``exec`` (T_a <= T_p) horizons."""

    obs: int
    pred: int
    exec: int


@dataclass(frozen=True)
class Optim:
    """Optimizer and learning-rate schedule."""

    name: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    schedule: str


@dataclass(frozen=True)
class Budget:
    """Fixed step budget, checkpoint cadence and logging (identical across families)."""

    max_steps: int
    batch_size: int
    ckpt_every: int
    val_every: int
    log_every: int
    num_threads: int
    grad_clip: float | None


@dataclass(frozen=True)
class RunConfig:
    """A fully resolved training-run configuration."""

    name: str
    family: str
    task: str
    seed: int
    horizons: Horizons
    model: dict[str, Any]
    loss: str
    optim: Optim
    train: Budget

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (the form that is hashed)."""
        return {"schema": SCHEMA, "name": self.name, "family": self.family, "task": self.task,
                "seed": self.seed,
                "horizons": vars(self.horizons).copy(), "model": copy.deepcopy(self.model),
                "loss": {"kind": self.loss},
                "optim": {**vars(self.optim), "betas": list(self.optim.betas)},
                "train": vars(self.train).copy()}

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return sha256_json(self.to_dict())


def _exact_keys(name: str, d: dict[str, Any], keys: set[str]) -> None:
    if not isinstance(d, dict) or set(d) != keys:
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise ValueError(f"config section {name!r}: keys {got} != {sorted(keys)}")


def from_dict(d: dict[str, Any]) -> RunConfig:
    """Validate and type a config dict (as loaded from YAML, after overrides)."""
    _exact_keys("<root>", d, {"schema", "name", "family", "task", "seed", *SECTIONS})
    if d["schema"] != SCHEMA:
        raise ValueError(f"schema {d['schema']!r} != {SCHEMA!r}")
    family = d["family"]
    if family not in FAMILIES:
        raise ValueError(f"family {family!r} not in {sorted(FAMILIES)}")
    if d["task"] not in ("lift", "can"):
        raise ValueError(f"task {d['task']!r} not in ('lift', 'can')")
    _exact_keys("horizons", d["horizons"], {"obs", "pred", "exec"})
    hz = Horizons(**{k: int(v) for k, v in d["horizons"].items()})
    if min(hz.obs, hz.pred, hz.exec) < 1 or hz.exec > hz.pred:
        raise ValueError(f"invalid horizons {hz}: need >= 1 and exec <= pred")
    model = config_dict(model_config(family, d["model"]))
    _exact_keys("loss", d["loss"], {"kind"})
    if d["loss"]["kind"] not in LOSSES:
        raise ValueError(f"loss {d['loss']['kind']!r} not in {LOSSES}")
    _exact_keys("optim", d["optim"], {"name", "lr", "betas", "weight_decay", "schedule"})
    o = d["optim"]
    optim = Optim(name=o["name"], lr=float(o["lr"]), betas=tuple(float(b) for b in o["betas"]),
                  weight_decay=float(o["weight_decay"]), schedule=o["schedule"])
    if optim.name not in OPTIMIZERS or optim.schedule not in SCHEDULES or len(optim.betas) != 2:
        raise ValueError(f"unsupported optimizer settings {optim}")
    _exact_keys("train", d["train"], {"max_steps", "batch_size", "ckpt_every", "val_every",
                                      "log_every", "num_threads", "grad_clip"})
    t = d["train"]
    budget = Budget(**{k: int(t[k]) for k in ("max_steps", "batch_size", "ckpt_every",
                                              "val_every", "log_every", "num_threads")},
                    grad_clip=None if t["grad_clip"] is None else float(t["grad_clip"]))
    if min(budget.max_steps, budget.batch_size, budget.ckpt_every, budget.val_every,
           budget.log_every, budget.num_threads) < 1:
        raise ValueError(f"budget values must be >= 1: {budget}")
    if budget.max_steps % budget.ckpt_every:
        raise ValueError("max_steps must be a multiple of ckpt_every (the last checkpoint is "
                         "the final step)")
    if not str(d["name"]).replace("_", "").isalnum():
        raise ValueError(f"name {d['name']!r} must be alphanumeric/underscore")
    return RunConfig(name=str(d["name"]), family=family, task=d["task"], seed=int(d["seed"]),
                     horizons=hz, model=model, loss=d["loss"]["kind"], optim=optim, train=budget)


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides (values parsed as YAML). Only existing keys may be set."""
    out = copy.deepcopy(d)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        node = out
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"override {path!r}: no such section {k!r}")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise KeyError(f"override {path!r}: no such key")
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    """Read a YAML run config and apply overrides."""
    raw = yaml.safe_load(Path(path).read_text())
    return from_dict(apply_overrides(raw, overrides or []))


def run_id(cfg: RunConfig, prefix: str) -> str:
    """Stable run ID: ``<prefix>_<name>_<task>_seed<s>_<config sha256[:8]>``."""
    if not prefix.replace("_", "").isalnum():
        raise ValueError(f"run prefix {prefix!r} must be alphanumeric/underscore")
    return f"{prefix}_{cfg.name}_{cfg.task}_seed{cfg.seed}_{cfg.sha256()[:8]}"
