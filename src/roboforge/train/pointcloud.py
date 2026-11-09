"""Point-cloud-conditioned diffusion policy of the Level 2 study: config, network, windows, trainer.

Session 9, MASTER_SPEC §7.3. The arm is matched to the frozen RGB diffusion arm
(``configs/models/rgb/diffusion_resnet18_lift.yaml``) in everything except the observation and
its encoder.

The training mechanics are the RGB arm's own code. :class:`PCTrainer` subclasses the frozen
:class:`roboforge.train.rgb.RGBTrainer` and inherits, unchanged:
* ``train_step``;
* ``save_cadence`` and ``save_resume``;
* ``run``.

So the sampler (``EpochStreamSampler(seed)``), the noise generator ``[seed, 3]``, AdamW, the EMA,
the checkpoint cadence and the resume rule are identical by construction. It overrides only
what touches the observation:
* construction;
* the offline validation loss;
* the checkpoint schema.

A crop generator ``[seed, 2]`` still exists and is saved, for the inherited code, but the point
cloud is not augmented, so it is never drawn from.

**Observation.** Each frame has one cloud (``float32 [N, 3]``, world metres; built by
:mod:`roboforge.data.pointcloud`) and the first 9 dims of the normalized low-dim vector. No
object state is used. Window ``i`` reads clouds ``window_indices(T_e, t, T_o, T_p).obs_idx`` of
its episode, clipped at the episode start with ``obs_valid`` false for the padded frame: the
RGB arm's rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from roboforge.data.pointcloud import PointCloudSpec
from roboforge.models.losses import masked_mse
from roboforge.models.point_encoder import PointCloudObservationEncoder
from roboforge.models.registry import build_model
from roboforge.models.vision import parameter_count
from roboforge.paths import REPO_ROOT
from roboforge.train.ema import EMA
from roboforge.train.rgb import RGBTrainer, set_deterministic

CONFIG_SCHEMA = "roboforge.pc_train_config/1"
CKPT_SCHEMA = "roboforge.pc_checkpoint/1"
ACTION_DIM = 7


@dataclass(frozen=True)
class PCConfig:
    """A parsed ``roboforge.pc_train_config/1`` file plus overrides."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path, overrides: dict[str, Any] | None = None) -> PCConfig:
        """Load ``path``; ``overrides`` maps dotted keys (``train.max_steps``) to values."""
        raw = yaml.safe_load(Path(path).read_text())
        if raw.get("schema") != CONFIG_SCHEMA:
            raise ValueError(f"{path}: not a {CONFIG_SCHEMA} file")
        for key, value in (overrides or {}).items():
            node = raw
            *parents, leaf = key.split(".")
            for p in parents:
                node = node[p]
            if leaf not in node:
                raise KeyError(f"override {key!r}: no such key")
            node[leaf] = value
        return cls(raw)

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def sha256(self) -> str:
        """Hash of the canonical JSON form (the run ID uses its first 8 hex digits)."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()

    def run_id(self, prefix: str) -> str:
        """``<prefix>_<name>_<task>_seed<s>_<sha[:8]>``, as Level 1 and the RGB study."""
        return f"{prefix}_{self.raw['name']}_{self.raw['task']}_seed{self.raw['seed']}_" \
               f"{self.sha256()[:8]}"

    @property
    def cloud_spec(self) -> PointCloudSpec:
        """The observation's :class:`PointCloudSpec`."""
        return PointCloudSpec.from_dict(self.raw["observation"]["cloud"])


class PCPolicyNet(nn.Module):
    """Point encoder + the Level 1 diffusion head; normalized actions in and out."""

    def __init__(self, cfg: PCConfig) -> None:
        super().__init__()
        o, e, h = cfg["observation"], cfg["encoder"], cfg["horizons"]
        if e["kind"] != "dp3" or e.get("augmentation", "none") != "none":
            raise ValueError(f"unsupported encoder {e}")
        lo, hi = o["proprio_dims"]
        spec = cfg.cloud_spec
        self.cameras = tuple(spec.cameras)
        self.proprio_slice = (int(lo), int(hi))
        self.enc = PointCloudObservationEncoder(hi - lo, spec.centre.tolist(),
                                                spec.half_extent.tolist(), e["widths"],
                                                e["out_dim"])
        self.net = build_model(cfg["head_family"], cfg["model"], self.enc.out_dim, ACTION_DIM,
                               h["obs"], h["pred"])
        self.stochastic = bool(getattr(self.net, "stochastic", False))

    def features(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Per-frame features ``[B, T_o, 64 + 9]``."""
        return self.enc(points, proprio)

    def loss(self, b: dict[str, torch.Tensor], crop_gen: torch.Generator | None,
             noise_gen: torch.Generator | None) -> torch.Tensor:
        """Training loss on one batch (``crop_gen`` is unused: no augmentation)."""
        f = self.features(b["points"], b["proprio"])
        if self.stochastic:
            return self.net.training_loss(f, b["obs_valid"], b["future_action"],
                                          b["valid_mask"], noise_gen)
        return masked_mse(self.net(f, b["obs_valid"]), b["future_action"], b["valid_mask"])

    @torch.no_grad()
    def predict(self, points: torch.Tensor, proprio: torch.Tensor, valid: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Normalized action chunk ``[B, T_p, A]`` (DDIM for diffusion, CPU)."""
        f = self.features(points, proprio)
        if self.stochastic:
            return self.net.sample(f, valid, generator)
        return self.net(f, valid)


def build_pc_policy(cfg: PCConfig) -> PCPolicyNet:
    """Seeded construction (``torch.manual_seed(seed)``, then encoder, then head)."""
    torch.manual_seed(int(cfg["seed"]))
    return PCPolicyNet(cfg)


@dataclass
class PCWindows:
    """All windows of one split: low-dim windows plus per-window global cloud indices."""

    proprio: torch.Tensor        # [N, T_o, P] float32 normalized
    obs_valid: torch.Tensor      # [N, T_o] bool
    future_action: torch.Tensor  # [N, T_p, A] float32 normalized
    valid_mask: torch.Tensor     # [N, T_p] bool
    frame_idx: torch.Tensor      # [N, T_o] int64 into ``points``
    points: torch.Tensor         # [F, n_points, 3] float32 (all frames of the split)

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def to(self, device: torch.device) -> PCWindows:
        """Every tensor on ``device``."""
        return PCWindows(self.proprio.to(device), self.obs_valid.to(device),
                         self.future_action.to(device), self.valid_mask.to(device),
                         self.frame_idx.to(device), self.points.to(device))

    def batch(self, idx: torch.Tensor, cameras: Any = None) -> dict[str, torch.Tensor]:
        """Windows at ``idx`` with clouds ``[B, T_o, n_points, 3]`` (``cameras`` is unused)."""
        return {"points": self.points[self.frame_idx[idx]], "proprio": self.proprio[idx],
                "obs_valid": self.obs_valid[idx], "future_action": self.future_action[idx],
                "valid_mask": self.valid_mask[idx]}


def load_pc_windows(cfg: PCConfig, split: str, pc: dict[str, Any]) -> PCWindows:
    """Windows of ``split`` from the committed low-dim data and the loaded clouds."""
    from roboforge.data.sequence import SequenceDataset
    from roboforge.data.windows import window_indices
    from roboforge.train.data import WindowTensors

    if PointCloudSpec.from_dict(pc["spec"]) != cfg.cloud_spec:
        raise ValueError("point-cloud dataset was built with another spec than the config's")
    h, o = cfg["horizons"], cfg["observation"]
    ds = SequenceDataset.from_committed(cfg["task"], split, obs_horizon=h["obs"],
                                        pred_horizon=h["pred"])
    w = WindowTensors.from_dataset(ds)
    demos = sorted(set(w.episode_id), key=lambda d: int(d.split("_")[1]))
    offset, lengths, offsets = 0, {}, {}
    for d in demos:
        n = len(pc["points"][d])
        lengths[d], offsets[d] = n, offset
        offset += n
    fi = np.empty((len(w), h["obs"]), dtype=np.int64)
    for i, (d, t) in enumerate(zip(w.episode_id, w.start_t.tolist(), strict=True)):
        fi[i] = offsets[d] + window_indices(lengths[d], int(t), h["obs"], h["pred"]).obs_idx
    lo, hi = o["proprio_dims"]
    points = torch.from_numpy(np.concatenate([pc["points"][d] for d in demos]))
    return PCWindows(w.obs_history[..., lo:hi].contiguous(), w.obs_valid, w.future_action,
                     w.valid_mask, torch.from_numpy(fi), points)


class PCTrainer(RGBTrainer):
    """One point-cloud run: the RGB arm's trainer with the point-cloud observation."""

    def __init__(self, cfg: PCConfig, run_dir: Path, train: PCWindows, valid: PCWindows,
                 context: dict[str, Any], device: str) -> None:  # noqa: D107
        from roboforge.train.data import EpochStreamSampler

        self.cfg, self.run_dir, self.context = cfg, Path(run_dir), context
        self.dev = torch.device(device)
        self.flags = set_deterministic(cfg) if self.dev.type == "cuda" else {}
        seed = int(cfg["seed"])
        self.model = build_pc_policy(cfg).to(self.dev)
        o = cfg["optim"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["lr"],
                                     betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
        m = cfg["model"]
        self.ema = (EMA(self.model, power=m["ema_power"], max_value=m["ema_max_decay"])
                    if m.get("ema") else None)
        # Kept for the inherited resume code; no augmentation ever draws from it.
        self.crop_gen = torch.Generator(device=self.dev)
        self.crop_gen.manual_seed(int(np.random.SeedSequence([seed, 2]).generate_state(1)[0]))
        self.noise_gen = torch.Generator(device=self.dev)
        self.noise_gen.manual_seed(int(np.random.SeedSequence([seed, 3]).generate_state(1)[0]))
        self.train, self.valid = train.to(self.dev), valid.to(self.dev)
        self.sampler = EpochStreamSampler(len(train), int(cfg["train"]["batch_size"]), seed)
        self.cameras = self.model.cameras
        self.step = 0

    @torch.no_grad()
    def valid_loss(self, batch: int = 256) -> float:
        """Offline loss on every validation window (logged only; never selects)."""
        net = self.ema.averaged_model if self.ema is not None else self.model
        net.eval()
        g = torch.Generator(device=self.dev).manual_seed(0)
        tot, n = 0.0, 0
        for s in range(0, len(self.valid), batch):
            idx = torch.arange(s, min(s + batch, len(self.valid)), device=self.dev)
            b = self.valid.batch(idx)
            f = net.features(b["points"], b["proprio"])
            if net.stochastic:
                loss = net.net.training_loss(f, b["obs_valid"], b["future_action"],
                                             b["valid_mask"], g)
            else:
                loss = masked_mse(net.net(f, b["obs_valid"]), b["future_action"],
                                  b["valid_mask"])
            tot += float(loss) * len(idx)
            n += len(idx)
        return tot / n

    def _base_payload(self) -> dict[str, Any]:
        return {"schema": CKPT_SCHEMA, "run_id": self.context["run_id"],
                "family": self.cfg["family"], "task": self.cfg["task"], "step": self.step,
                "config": self.cfg.raw, "config_sha256": self.cfg.sha256(),
                "normalization": self.context["normalization"],
                "hashes": self.context["hashes"],
                "param_count": parameter_count(self.model)}

    def load_resume(self) -> int:
        """Restore ``resume/latest.pt``; returns the step it resumes from."""
        from roboforge.train.rgb import _optimizer_to

        p = torch.load(self.run_dir / "resume" / "latest.pt", map_location="cpu",
                       weights_only=True)
        if p["schema"] != CKPT_SCHEMA or p["config_sha256"] != self.cfg.sha256():
            raise ValueError("resume checkpoint belongs to another config")
        self.model.load_state_dict(p["model"])
        self.opt.load_state_dict(p["optimizer"])
        _optimizer_to(self.opt, self.dev)
        self.sampler.load_state_dict(p["sampler"])
        self.crop_gen.set_state(p["crop_gen"])
        self.noise_gen.set_state(p["noise_gen"])
        if self.ema is not None:
            self.ema.load_state_dict(p["ema_state"])
            self.ema.averaged_model.to(self.dev)
        self.step = int(p["step"])
        return self.step


def config_path() -> Path:
    """The committed config of the point-cloud arm."""
    return REPO_ROOT / "configs" / "models" / "pointcloud" / "diffusion_dp3_lift.yaml"


__all__ = ["CKPT_SCHEMA", "CONFIG_SCHEMA", "PCConfig", "PCPolicyNet", "PCTrainer", "PCWindows",
           "build_pc_policy", "config_path", "load_pc_windows"]
