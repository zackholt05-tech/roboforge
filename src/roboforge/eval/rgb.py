"""Closed-loop evaluation of RGB-study checkpoints (Session 8; D-078, the RGB freeze).

Level 1's runner pieces are reused read-only: ``run_episode``, ``verify_record``,
``episode_stats``, the suite loaders and the run-ID rule. Only the policy is new.

:class:`RGBClosedLoopPolicy`:
* **Images.** It renders the two cameras at 84² through ``RoboForgeEnv.render``, the D-074
  path. The first render of the env's renderer is discarded as a warm-up.
* **Selective rendering.** It renders only the frames a decision reads. A decision at frame
  ``n`` reads frames ``n − 1`` and ``n`` (clipped at 0, ``obs_valid`` as in training). After a
  decision at ``d``, the next comes at ``d + T_a``. Frames ``d + T_a − 1`` and ``d + T_a`` are
  rendered when observed, and nothing else.
* **Proprioception.** The first 9 dims of the normalized low-dim vector, with the train
  statistics stored in the checkpoint.
* **Network.** On the CPU with one torch thread, centre crop:
  * diffusion uses the EMA weights, DDIM-10, and Level 1's per-episode noise seed rule;
  * BC uses the trained weights.
* **Actions.** The first ``T_a = 8`` actions of each chunk, decoded by Level 1's
  ``ActionCodec``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.data.normalization import ActionCodec, Normalizer
from roboforge.data.windows import window_indices
from roboforge.eval.learned import CONTRACT_HASHES, NOISE_SEED_RULE, noise_seed
from roboforge.eval.policies import PolicyInfo
from roboforge.provenance import sha256_file
from roboforge.train.rgb import CKPT_SCHEMA, RGBConfig, RGBPolicyNet

SCHEMA = "roboforge.rgb_validation_eval/1"


def load_rgb_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    """A hash-verified RGB cadence checkpoint (``weights_only``)."""
    sha = sha256_file(path)
    if sha != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {sha} != expected {expected_sha256}")
    p = torch.load(path, map_location="cpu", weights_only=True)
    if p.get("schema") != CKPT_SCHEMA:
        raise ValueError(f"{path}: not a {CKPT_SCHEMA} checkpoint")
    return p


class RGBClosedLoopPolicy:
    """An RGB-study checkpoint as a closed-loop policy in the normalized action space."""

    def __init__(self, payload: dict[str, Any], checkpoint_sha256: str, *,
                 policy_seed: int = 0, repeat: int = 0) -> None:
        torch.set_num_threads(1)
        self.cfg = RGBConfig(payload["config"])
        self.model = RGBPolicyNet(self.cfg)
        weights = payload["ema_model"] if payload.get("ema_model") is not None \
            else payload["model"]
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.stochastic = self.model.stochastic
        o, h = self.cfg["observation"], self.cfg["horizons"]
        self.cameras, self.size = tuple(o["cameras"]), int(o["image_size"])
        self.lo, self.hi = (int(x) for x in o["proprio_dims"])
        self.obs_horizon, self.exec_horizon = int(h["obs"]), int(h["exec"])
        self.task, self.hashes = payload["task"], dict(payload["hashes"])
        self.obs_normalizer = Normalizer.from_dict(payload["normalization"]["obs"])
        self.action_normalizer = Normalizer.from_dict(payload["normalization"]["action"])
        self.policy_seed, self.repeat = int(policy_seed), int(repeat)
        self.info = PolicyInfo(
            name=self.cfg["name"], kind="learned", learned=True, autonomous=True,
            action_space="normalized",
            label=(f"learned {self.cfg['family']} policy (RGB study: two 84² cameras + 9-D "
                   f"proprioception, no object state); checkpoint {checkpoint_sha256[:12]} at "
                   f"step {payload['step']}"),
            params={"run_id": payload["run_id"], "family": self.cfg["family"],
                    "global_step": int(payload["step"]),
                    "checkpoint_sha256": checkpoint_sha256,
                    "config_sha256": payload["config_sha256"],
                    "param_count": payload["param_count"],
                    "inference_network": "ema" if payload.get("ema_model") is not None
                    else "trained", "num_threads": torch.get_num_threads(), "device": "cpu",
                    "cameras": list(self.cameras), "image_size": self.size,
                    "torch_version": str(torch.__version__)})
        if self.stochastic:
            self.info.params.update(policy_seed=self.policy_seed, repeat=self.repeat,
                                    noise_seed_rule=NOISE_SEED_RULE)
        self.env: Any = None
        self._warm_for: int | None = None
        self.n_renders = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._prop: list[np.ndarray] = []
        self._img: dict[int, np.ndarray] = {}
        self._next_decision = 0
        self.n_decisions = 0
        self._gen: torch.Generator | None = None
        self._episode: dict[str, Any] | None = None

    def bind(self, env: Any) -> None:
        """The env whose current state is rendered (called once per env by the runner)."""
        self.env = env

    def codec(self, action_spec: Any) -> ActionCodec:
        """Decoder from normalized model actions to clipped simulator actions."""
        return ActionCodec(self.action_normalizer, np.array(action_spec.low, dtype=np.float64),
                           np.array(action_spec.high, dtype=np.float64))

    def reset_episode(self, episode: Any) -> None:
        """Clear the history; refuse another task or data contract; seed the sampler."""
        if episode.task != self.task:
            raise ValueError(f"policy trained on {self.task!r}, episode is {episode.task!r}")
        for k in CONTRACT_HASHES:
            if getattr(episode, k) != self.hashes[k]:
                raise ValueError(f"episode {k} differs from the checkpoint's")
        self._reset_state()
        if self.stochastic:
            sid = episode.initial_state["state_id"]
            seed = noise_seed(self.policy_seed, sid, self.repeat)
            self._gen = torch.Generator().manual_seed(seed)
            self._episode = {"state_id": sid, "policy_seed": self.policy_seed,
                             "repeat": self.repeat, "noise_seed": seed,
                             "noise_seed_rule": NOISE_SEED_RULE}

    def episode_info(self) -> dict[str, Any] | None:
        """Per-episode sampling provenance (stochastic policies), stored in the record."""
        return None if self._episode is None else dict(self._episode)

    def _render(self) -> np.ndarray:
        shadow_key = id(getattr(self.env, "_shadow", None))
        if self._warm_for != shadow_key or getattr(self.env, "_shadow", None) is None:
            self.env.render(self.cameras[0], self.size, self.size)  # warm-up, discarded
            self._warm_for = id(self.env._shadow)
        self.n_renders += 1
        return np.stack([self.env.render(c, self.size, self.size) for c in self.cameras])

    def observe(self, obs: Any) -> None:
        """Record one frame (after reset and after every step); render if a decision reads it."""
        if self.env is None:
            raise RuntimeError("bind(env) before the episode")
        k = len(self._prop)
        v = self.obs_normalizer.normalize(np.asarray(obs.vector).astype(np.float32))
        self._prop.append(v.astype(np.float32)[self.lo:self.hi])
        if k in (self._next_decision - 1, self._next_decision):
            self._img[k] = self._render()

    def act(self, obs: Any, trace: Any = None) -> np.ndarray:
        """First ``T_a`` normalized actions of the predicted chunk, ``[T_a, A]`` float64."""
        n = len(self._prop)
        if n - 1 != self._next_decision:
            raise RuntimeError(f"decision at frame {n - 1}, expected {self._next_decision}")
        w = window_indices(n, n - 1, self.obs_horizon, 1)
        missing = [int(i) for i in w.obs_idx if int(i) not in self._img]
        if missing:
            raise RuntimeError(f"frames {missing} were not rendered")
        img = torch.from_numpy(np.stack([self._img[int(i)] for i in w.obs_idx]))[None]
        prop = torch.from_numpy(np.stack([self._prop[int(i)] for i in w.obs_idx]))[None]
        valid = torch.from_numpy(w.obs_valid)[None]
        out = self.model.predict(img, prop, valid, self._gen)[0]
        self.n_decisions += 1
        self._next_decision = n - 1 + self.exec_horizon
        for i in [i for i in self._img if i < n - 1]:  # keep memory flat
            del self._img[i]
        return out[: self.exec_horizon].numpy().astype(np.float64)


def run_rgb_states(policy: RGBClosedLoopPolicy, states: list[Any], eval_id: str,
                   out_dir: Path, task: str, n_verify: int) -> dict[str, Any]:
    """Level 1's ``run_states`` with the env bound to the policy (it renders that env)."""
    from roboforge.eval.rollout import run_episode, verify_record
    from roboforge.eval.validation import episode_stats
    from roboforge.sim.env_adapter import RoboForgeEnv

    env = RoboForgeEnv(task)
    policy.bind(env)
    codec = policy.codec(env.action_spec)
    episodes, verified = [], []
    t0 = time.perf_counter()
    try:
        for i, init in enumerate(states):
            rec = run_episode(env, policy, init, run_id=eval_id, episode_index=i, codec=codec)
            rec.save(out_dir / "episodes" / init.state_id)
            if i < n_verify:
                verified.append({"state_id": init.state_id, **verify_record(rec, init)})
            episodes.append({"state_id": init.state_id,
                             "physics_sha256": rec.meta["physics_sha256"],
                             **({"noise_seed": rec.meta["policy_episode"]["noise_seed"]}
                                if policy.stochastic else {}),
                             **episode_stats(rec, codec)})
        rules = {"horizon": env.horizon, "termination": env.termination,
                 "hold_steps": env.hold_steps}
        dim = env.action_dim
    finally:
        env.close()
    return {"episodes": episodes, "verified": verified, "wall_s": time.perf_counter() - t0,
            "episode_rules": rules, "action_dim": dim, "renders": policy.n_renders}


def evaluate_rgb_checkpoint(run_dir: Path, step: int, *, out_root: Path,
                            n_states: int | None = None, n_verify: int = 3) -> dict[str, Any]:
    """One RGB checkpoint on the Lift validation suite (the only suite this allows)."""
    import json

    from roboforge.eval.validation import (
        VALIDATION,
        _summarize,
        canonical_platform_id,
        check_suite_allowed,
        eval_run_id,
    )
    from roboforge.paths import repo_relative
    from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
    from roboforge.sim.eval_states import load_suite, read_manifest
    from roboforge.train.run import write_manifest

    man = json.loads((Path(run_dir) / "manifest.json").read_text())
    ck = {c["step"]: c for c in man["checkpoints"]}
    if step not in ck:
        raise KeyError(f"{man['run_id']} has no checkpoint at step {step}: {sorted(ck)}")
    payload = load_rgb_checkpoint(Path(run_dir) / ck[step]["path"], ck[step]["sha256"])
    policy = RGBClosedLoopPolicy(payload, ck[step]["sha256"])
    task = man["task"]
    check_suite_allowed(task, VALIDATION)
    states = load_suite(task, VALIDATION)
    partial = n_states is not None and n_states < len(states)
    states = states[:n_states] if partial else states
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    eval_id = eval_run_id(man["run_id"], step, platform["platform_id"],
                          stochastic=policy.stochastic,
                          n_partial=len(states) if partial else None)
    out_dir = Path(out_root) / eval_id
    out_dir.mkdir(parents=True, exist_ok=False)
    run = run_rgb_states(policy, states, eval_id, out_dir, task, n_verify)
    suite = read_manifest(task, VALIDATION)
    summary = {
        "schema": SCHEMA, "eval_run_id": eval_id,
        "kind": "closed-loop validation evaluation (RGB study)",
        "created_utc": datetime.now(tz=UTC).isoformat(), "train_run_id": man["run_id"],
        "checkpoint": {"step": step, **ck[step]}, "policy": policy.info.to_dict(),
        "task": task, "suite": VALIDATION, "partition": suite["partition"],
        "suite_manifest_content_sha256": suite["content_sha256"],
        "stochastic_policy": ({"policy_seed": 0, "repeat": 0, "samples_per_decision": 1}
                              if policy.stochastic else None),
        "platform": platform, "canonical_platform": canonical,
        "label": ("canonical platform (D-017)" if canonical
                  else "DEVELOPMENT ONLY: native platform, never pooled with canonical results"),
        "eligible_for_selection": bool(canonical and not partial), "partial": partial,
        "git": git_revision(), "environment": environment_snapshot(),
        "episode_rules": run["episode_rules"], **_summarize(run["episodes"], run["action_dim"]),
        "renders": run["renders"], "wall_s": run["wall_s"],
        "verify_record": run["verified"],
        "verify_passed": all(v["passed"] for v in run["verified"]),
        "records_dir": repo_relative(out_dir), "episodes": run["episodes"]}
    write_manifest(out_dir / "manifest.json", summary)
    return summary


__all__ = ["SCHEMA", "RGBClosedLoopPolicy", "evaluate_rgb_checkpoint", "load_rgb_checkpoint",
           "run_rgb_states"]
