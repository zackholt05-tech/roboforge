"""RGB-conditioned policies of the Session 8 study: config, network, windows, trainer.

D-078; ``docs/RGB_EXPERIMENT_FREEZE.md``; RESEARCH_PROTOCOL §13 amendment of 2026-11-08.
Level 1 code is reused read-only: the head networks (``build_model``), the windows
(``SequenceDataset`` → ``WindowTensors``), the sampler (``EpochStreamSampler``), the loss
(``masked_mse``) and the EMA. Nothing here edits a frozen module.

**Observation.** Each frame has two camera images (``uint8`` 84², random crop 76 while
training, centre crop otherwise) and the first 9 dims of the normalized low-dim vector:
``robot0_eef_pos``, ``robot0_eef_quat``, ``robot0_gripper_qpos``. No object state is used.
Window ``i`` of the low-dim dataset (episode ``e``, decision step ``t``) reads frames
``window_indices(T_e, t, T_o, T_p).obs_idx`` of episode ``e``, which also clip at the
episode start. ``obs_valid`` marks the padded ones, exactly as for the proprioception.

**Shared encoder.** ``torch.manual_seed(seed)`` is called, then the encoder is built, then
the head. Both families therefore start from the same encoder weights for a given seed (a
test checks this).

**Separate generators.** Crops come from a generator seeded ``[seed, 2]`` and diffusion noise
from ``[seed, 3]``. Since the sampler sees the same batch stream (``EpochStreamSampler(seed)``),
both families get identical batches and identical crops.

**Checkpoints** (schema ``roboforge.rgb_checkpoint/1``):
* the cadence file ``checkpoints/step_<k>.pt`` holds the trained weights, the EMA weights
  (diffusion), the config, the normalization, the data hashes and the step;
* the single resume file ``resume/latest.pt`` adds optimizer, sampler and generator states,
  so resumed training equals uninterrupted training within a host (the amendment's claim);
* nothing is ever overwritten except ``resume/latest.pt``, which is replaced atomically. A
  cadence file left by a kill before its resume file (so never listed in the manifest) is
  renamed ``step_<k>.orphan<n>.pt`` when the restart recomputes that step.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from roboforge.models.losses import masked_mse
from roboforge.models.registry import build_model
from roboforge.models.vision import RGBObservationEncoder, parameter_count
from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file
from roboforge.train.ema import EMA

CONFIG_SCHEMA = "roboforge.rgb_train_config/1"
CKPT_SCHEMA = "roboforge.rgb_checkpoint/1"
ACTION_DIM = 7


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class RGBConfig:
    """A parsed ``roboforge.rgb_train_config/1`` file plus overrides."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path, overrides: dict[str, Any] | None = None) -> RGBConfig:
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
        """``<prefix>_<name>_<task>_seed<s>_<sha[:8]>``, as Level 1."""
        return f"{prefix}_{self.raw['name']}_{self.raw['task']}_seed{self.raw['seed']}_" \
               f"{self.sha256()[:8]}"


# ----------------------------------------------------------------------------- network
class RGBPolicyNet(nn.Module):
    """Encoder (per camera) + Level 1 head; normalized actions in and out."""

    def __init__(self, cfg: RGBConfig) -> None:
        super().__init__()
        o, e, h = cfg["observation"], cfg["encoder"], cfg["horizons"]
        lo, hi = o["proprio_dims"]
        self.cameras = tuple(o["cameras"])
        self.proprio_slice = (int(lo), int(hi))
        self.enc = RGBObservationEncoder(self.cameras, hi - lo, e["trunk"], o["image_size"],
                                         o["crop"], e["feature_dim"], e["num_kp"])
        self.net = build_model(cfg["head_family"], cfg["model"], self.enc.out_dim, ACTION_DIM,
                               h["obs"], h["pred"])
        self.stochastic = bool(getattr(self.net, "stochastic", False))

    def features(self, images: torch.Tensor, proprio: torch.Tensor, *, train: bool,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Per-frame features ``[B, T_o, n_cam · 64 + 9]``."""
        return self.enc(images, proprio, train=train, generator=generator)

    def loss(self, b: dict[str, torch.Tensor], crop_gen: torch.Generator,
             noise_gen: torch.Generator | None) -> torch.Tensor:
        """Training loss on one batch (random crops)."""
        f = self.features(b["images"], b["proprio"], train=True, generator=crop_gen)
        if self.stochastic:
            return self.net.training_loss(f, b["obs_valid"], b["future_action"],
                                          b["valid_mask"], noise_gen)
        return masked_mse(self.net(f, b["obs_valid"]), b["future_action"], b["valid_mask"])

    @torch.no_grad()
    def predict(self, images: torch.Tensor, proprio: torch.Tensor, valid: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Normalized action chunk ``[B, T_p, A]`` (centre crops; DDIM for diffusion, CPU)."""
        f = self.features(images, proprio, train=False)
        if self.stochastic:
            return self.net.sample(f, valid, generator)
        return self.net(f, valid)


def build_policy(cfg: RGBConfig) -> RGBPolicyNet:
    """Seeded construction: the encoder first, so it is identical across families."""
    torch.manual_seed(int(cfg["seed"]))
    return RGBPolicyNet(cfg)


# ----------------------------------------------------------------------------- data
@dataclass
class RGBWindows:
    """All windows of one split: low-dim windows plus per-window global frame indices."""

    proprio: torch.Tensor        # [N, T_o, P] float32 normalized
    obs_valid: torch.Tensor      # [N, T_o] bool
    future_action: torch.Tensor  # [N, T_p, A] float32 normalized
    valid_mask: torch.Tensor     # [N, T_p] bool
    frame_idx: torch.Tensor      # [N, T_o] int64 into ``images[cam]``
    images: dict[str, torch.Tensor]  # camera -> [F, H, W, 3] uint8 (all frames of the split)

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def to(self, device: torch.device) -> RGBWindows:
        """Every tensor on ``device``."""
        return RGBWindows(self.proprio.to(device), self.obs_valid.to(device),
                          self.future_action.to(device), self.valid_mask.to(device),
                          self.frame_idx.to(device),
                          {c: v.to(device) for c, v in self.images.items()})

    def batch(self, idx: torch.Tensor, cameras: tuple[str, ...]) -> dict[str, torch.Tensor]:
        """Windows at ``idx`` with images ``[B, T_o, n_cam, H, W, 3]``."""
        fi = self.frame_idx[idx]
        return {"images": torch.stack([self.images[c][fi] for c in cameras], dim=2),
                "proprio": self.proprio[idx], "obs_valid": self.obs_valid[idx],
                "future_action": self.future_action[idx], "valid_mask": self.valid_mask[idx]}


def load_rgb_windows(cfg: RGBConfig, split: str, rgb: dict[str, Any]) -> RGBWindows:
    """Windows of ``split`` from the committed low-dim data and the loaded RGB frames."""
    from roboforge.data.sequence import SequenceDataset
    from roboforge.data.windows import window_indices
    from roboforge.train.data import WindowTensors

    h, o = cfg["horizons"], cfg["observation"]
    ds = SequenceDataset.from_committed(cfg["task"], split, obs_horizon=h["obs"],
                                        pred_horizon=h["pred"])
    w = WindowTensors.from_dataset(ds)
    demos = sorted(set(w.episode_id), key=lambda d: int(d.split("_")[1]))
    offset, lengths, offsets = 0, {}, {}
    for d in demos:
        n = len(rgb["frames"][d][o["cameras"][0]])
        lengths[d], offsets[d] = n, offset
        offset += n
    fi = np.empty((len(w), h["obs"]), dtype=np.int64)
    for i, (d, t) in enumerate(zip(w.episode_id, w.start_t.tolist(), strict=True)):
        fi[i] = offsets[d] + window_indices(lengths[d], int(t), h["obs"], h["pred"]).obs_idx
    lo, hi = o["proprio_dims"]
    images = {c: torch.from_numpy(np.concatenate([rgb["frames"][d][c] for d in demos]))
              for c in o["cameras"]}
    return RGBWindows(w.obs_history[..., lo:hi].contiguous(), w.obs_valid, w.future_action,
                      w.valid_mask, torch.from_numpy(fi), images)


# ----------------------------------------------------------------------------- trainer
def set_deterministic(cfg: RGBConfig) -> dict[str, Any]:
    """Apply the amendment's flags (CUDA only; on CPU training is deterministic anyway)."""
    dev = cfg["device"]
    if dev["kind"] == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", dev["cublas_workspace_config"])
        torch.use_deterministic_algorithms(bool(dev["deterministic"]))
        torch.backends.cudnn.deterministic = bool(dev["deterministic"])
        torch.backends.cudnn.benchmark = False
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def state_digest(module: nn.Module) -> str:
    """SHA-256 of a module's state (names and float bytes, host order)."""
    hsh = hashlib.sha256()
    for name, t in module.state_dict().items():
        hsh.update(name.encode())
        hsh.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return hsh.hexdigest()


def _atomic_save(obj: dict[str, Any], path: Path, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; RGB checkpoints are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


class RGBTrainer:
    """One RGB run: deterministic steps, cadence checkpoints, resume, deadline stop."""

    def __init__(self, cfg: RGBConfig, run_dir: Path, train: RGBWindows, valid: RGBWindows,
                 context: dict[str, Any], device: str) -> None:
        from roboforge.train.data import EpochStreamSampler

        self.cfg, self.run_dir, self.context = cfg, Path(run_dir), context
        self.dev = torch.device(device)
        self.flags = set_deterministic(cfg) if self.dev.type == "cuda" else {}
        seed = int(cfg["seed"])
        self.model = build_policy(cfg).to(self.dev)
        o = cfg["optim"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["lr"],
                                     betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
        m = cfg["model"]
        self.ema = (EMA(self.model, power=m["ema_power"], max_value=m["ema_max_decay"])
                    if m.get("ema") else None)
        self.crop_gen = torch.Generator(device=self.dev)
        self.crop_gen.manual_seed(int(np.random.SeedSequence([seed, 2]).generate_state(1)[0]))
        self.noise_gen = torch.Generator(device=self.dev)
        self.noise_gen.manual_seed(int(np.random.SeedSequence([seed, 3]).generate_state(1)[0]))
        self.train, self.valid = train.to(self.dev), valid.to(self.dev)
        self.sampler = EpochStreamSampler(len(train), int(cfg["train"]["batch_size"]), seed)
        self.cameras = self.model.cameras
        self.step = 0

    # -- one step (mirrors Trainer.step; rgb_bench.Bench.step timed the same) --
    def train_step(self) -> dict[str, float]:
        """One optimizer step on the next batch."""
        idx = torch.as_tensor(self.sampler.next_batch(), device=self.dev)
        b = self.train.batch(idx, self.cameras)
        self.model.train()
        loss = self.model.loss(b, self.crop_gen, self.noise_gen)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {self.step + 1}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.get_total_norm([p.grad for p in self.model.parameters()])
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step + 1}")
        self.opt.step()
        if self.ema is not None:
            self.ema.step(self.model)
        self.step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(gnorm.detach())}

    @torch.no_grad()
    def valid_loss(self, batch: int = 256) -> float:
        """Offline loss on every validation window (logged only; never selects).

        Centre crops; diffusion uses a fixed noise generator seeded 0 on the device."""
        net = self.ema.averaged_model if self.ema is not None else self.model
        net.eval()
        g = torch.Generator(device=self.dev).manual_seed(0)
        tot, n = 0.0, 0
        for s in range(0, len(self.valid), batch):
            idx = torch.arange(s, min(s + batch, len(self.valid)), device=self.dev)
            b = self.valid.batch(idx, self.cameras)
            f = net.features(b["images"], b["proprio"], train=False)
            if net.stochastic:
                loss = net.net.training_loss(f, b["obs_valid"], b["future_action"],
                                             b["valid_mask"], g)
            else:
                loss = masked_mse(net.net(f, b["obs_valid"]), b["future_action"],
                                  b["valid_mask"])
            tot += float(loss) * len(idx)
            n += len(idx)
        return tot / n

    # -- checkpoints --
    def _base_payload(self) -> dict[str, Any]:
        return {"schema": CKPT_SCHEMA, "run_id": self.context["run_id"],
                "family": self.cfg["family"], "task": self.cfg["task"], "step": self.step,
                "config": self.cfg.raw, "config_sha256": self.cfg.sha256(),
                "normalization": self.context["normalization"],
                "hashes": self.context["hashes"],
                "param_count": parameter_count(self.model)}

    def save_cadence(self) -> dict[str, Any]:
        """Inference checkpoint at the current step (trained + EMA weights)."""
        p = {**self._base_payload(), "model": self.model.state_dict(),
             "ema_model": (self.ema.averaged_model.state_dict() if self.ema else None),
             "ema_config": (self.ema.config() if self.ema else None)}
        p = _to_cpu(p)
        path = self.run_dir / "checkpoints" / f"step_{self.step:07d}.pt"
        if path.exists():
            # A kill between this file and the resume file of the same step makes a restart
            # recompute the step. The manifest (written after both) never listed the old
            # file, so it is kept aside, never reused.
            k = 0
            while path.with_suffix(f".orphan{k}.pt").exists():
                k += 1
            path.rename(path.with_suffix(f".orphan{k}.pt"))
        sha = _atomic_save(p, path)
        return {"step": self.step, "path": path.relative_to(self.run_dir).as_posix(),
                "sha256": sha, "inference_network": "ema" if self.ema else "trained",
                "inference_digest": state_digest(self.ema.averaged_model if self.ema
                                                 else self.model)}

    def save_resume(self) -> str:
        """Full training state; replaces ``resume/latest.pt`` atomically."""
        p = {**self._base_payload(), "model": self.model.state_dict(),
             "optimizer": self.opt.state_dict(), "sampler": self.sampler.state_dict(),
             "crop_gen": self.crop_gen.get_state(), "noise_gen": self.noise_gen.get_state(),
             "ema_state": (self.ema.state_dict() if self.ema else None)}
        return _atomic_save(_to_cpu(p), self.run_dir / "resume" / "latest.pt", overwrite=True)

    def load_resume(self) -> int:
        """Restore ``resume/latest.pt``; returns the step it resumes from."""
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

    # -- the loop --
    def run(self, *, should_stop: Callable[[], bool] = lambda: False,
            stop_at_step: int | None = None,
            log: Callable[[dict[str, Any]], None] = lambda r: None,
            on_checkpoint: Callable[[dict[str, Any]], None] = lambda r: None) -> str:
        """Train until ``max_steps`` (``complete``), or until ``should_stop()`` (deadline or
        preemption signal) or ``stop_at_step`` (``interrupted``, after writing the resume
        file)."""
        t = self.cfg["train"]
        max_steps, every, log_every = int(t["max_steps"]), int(t["ckpt_every"]), int(t["log_every"])
        t0 = time.perf_counter()
        while self.step < max_steps:
            if should_stop() or (stop_at_step is not None and self.step >= stop_at_step):
                self.save_resume()
                return "interrupted"
            rec = self.train_step()
            if self.step % log_every == 0:
                log({"step": self.step, **rec, "wall_s": time.perf_counter() - t0})
            if self.step % every == 0:
                ck = self.save_cadence()
                ck["valid_loss"] = self.valid_loss()
                self.save_resume()
                on_checkpoint(ck)
        return "complete"


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def _optimizer_to(opt: torch.optim.Optimizer, dev: torch.device) -> None:
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and k != "step":
                st[k] = v.to(dev)


def config_path(family: str) -> Path:
    """Committed config of an RGB family (``bc_chunk`` or ``diffusion``)."""
    return REPO_ROOT / "configs" / "models" / "rgb" / f"{family}_resnet18_lift.yaml"


__all__ = ["CKPT_SCHEMA", "CONFIG_SCHEMA", "RGBConfig", "RGBPolicyNet", "RGBTrainer",
           "RGBWindows", "build_policy", "config_path", "load_rgb_windows", "set_deterministic",
           "state_digest"]
