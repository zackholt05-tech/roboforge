"""Closed-loop evaluation for the Level 2 study (Session 9): the point-cloud arm, and the
frozen RGB diffusion arm on the preregistered placement sets.

The runner is the RGB study's own: ``run_rgb_states`` from the frozen ``roboforge.eval.rgb``.
It binds the env to the policy, decodes with Level 1's ``ActionCodec``, records with
``run_episode`` and audits with ``verify_record``. So both arms roll out through one code path.
Only the point-cloud policy is new.

:class:`PCClosedLoopPolicy` mirrors ``RGBClosedLoopPolicy``:
* **Clouds.** A cloud is built from the env's current state and compiled model by
  :class:`roboforge.data.pointcloud.CloudMaker`, the same function that built the training
  clouds:
  * depth from both cameras, with sites hidden and one discarded warm-up per GL context;
  * calibration;
  * the workspace box;
  * farthest-point sampling.

  A new ``CloudMaker`` is made whenever the env's compiled model changes, at each episode's
  restore.
* **Selective rendering.** A decision at frame ``n`` reads frames ``n − 1`` and ``n``. Only
  those frames are rendered: frame 0, then ``d + T_a − 1`` and ``d + T_a`` after each
  decision ``d``.
* **Proprioception.** The first 9 dims of the normalized low-dim vector, with the train
  statistics stored in the checkpoint.
* **Network.** On the CPU with one torch thread: EMA weights, DDIM-10, and Level 1's
  per-episode noise seed rule (``policy_seed = 0``, ``repeat = 0``).
* **Actions.** The first ``T_a = 8`` actions of each chunk.

Suites: ``validation`` (selection) and the Lift robustness sets ``robustness_base`` and
``ood_cube_shift`` (``check_suite_allowed``). The held-out suite is never reachable from here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.data.normalization import ActionCodec, Normalizer
from roboforge.data.pointcloud import CloudMaker
from roboforge.data.windows import window_indices
from roboforge.eval.learned import CONTRACT_HASHES, NOISE_SEED_RULE, noise_seed
from roboforge.eval.policies import PolicyInfo
from roboforge.provenance import sha256_file
from roboforge.train.pointcloud import CKPT_SCHEMA, PCConfig, PCPolicyNet

SCHEMA = "roboforge.level2_eval/1"


def load_pc_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    """A hash-verified point-cloud cadence checkpoint (``weights_only``)."""
    sha = sha256_file(path)
    if sha != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {sha} != expected {expected_sha256}")
    p = torch.load(path, map_location="cpu", weights_only=True)
    if p.get("schema") != CKPT_SCHEMA:
        raise ValueError(f"{path}: not a {CKPT_SCHEMA} checkpoint")
    return p


class PCClosedLoopPolicy:
    """A Level 2 point-cloud checkpoint as a closed-loop policy in the normalized action space."""

    def __init__(self, payload: dict[str, Any], checkpoint_sha256: str, *,
                 policy_seed: int = 0, repeat: int = 0) -> None:
        torch.set_num_threads(1)
        self.cfg = PCConfig(payload["config"])
        self.spec = self.cfg.cloud_spec
        self.model = PCPolicyNet(self.cfg)
        weights = payload["ema_model"] if payload.get("ema_model") is not None \
            else payload["model"]
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.stochastic = self.model.stochastic
        o, h = self.cfg["observation"], self.cfg["horizons"]
        self.lo, self.hi = (int(x) for x in o["proprio_dims"])
        self.obs_horizon, self.exec_horizon = int(h["obs"]), int(h["exec"])
        self.task, self.hashes = payload["task"], dict(payload["hashes"])
        self.obs_normalizer = Normalizer.from_dict(payload["normalization"]["obs"])
        self.action_normalizer = Normalizer.from_dict(payload["normalization"]["action"])
        self.policy_seed, self.repeat = int(policy_seed), int(repeat)
        self.info = PolicyInfo(
            name=self.cfg["name"], kind="learned", learned=True, autonomous=True,
            action_space="normalized",
            label=(f"learned {self.cfg['family']} policy (Level 2, DP3-inspired: one "
                   f"{self.spec.n_points}-point world-frame cloud from the depth of two "
                   f"{self.spec.size}² cameras + 9-D proprioception, no object state); "
                   f"checkpoint {checkpoint_sha256[:12]} at step {payload['step']}"),
            params={"run_id": payload["run_id"], "family": self.cfg["family"],
                    "global_step": int(payload["step"]),
                    "checkpoint_sha256": checkpoint_sha256,
                    "config_sha256": payload["config_sha256"],
                    "param_count": payload["param_count"],
                    "inference_network": "ema" if payload.get("ema_model") is not None
                    else "trained", "num_threads": torch.get_num_threads(), "device": "cpu",
                    "cloud_spec_sha256": self.spec.sha256(),
                    "torch_version": str(torch.__version__)})
        if self.stochastic:
            self.info.params.update(policy_seed=self.policy_seed, repeat=self.repeat,
                                    noise_seed_rule=NOISE_SEED_RULE)
        self.env: Any = None
        self._maker: CloudMaker | None = None
        self._maker_model: Any = None
        self.n_renders = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._prop: list[np.ndarray] = []
        self._pts: dict[int, np.ndarray] = {}
        self._next_decision = 0
        self.n_decisions = 0
        self._gen: torch.Generator | None = None
        self._episode: dict[str, Any] | None = None

    def bind(self, env: Any) -> None:
        """The env whose current state is observed (called once per env by the runner)."""
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

    def _cloud(self) -> np.ndarray:
        model = self.env.model
        if self._maker is None or self._maker_model is not model:
            if self._maker is not None:
                self._maker.close()
            self._maker, self._maker_model = CloudMaker(model, self.spec), model
        self.n_renders += 1
        cloud, _ = self._maker.cloud(self.env.current_state())
        return cloud

    def observe(self, obs: Any) -> None:
        """Record one frame (after reset and after every step); build a cloud if a decision
        reads it."""
        if self.env is None:
            raise RuntimeError("bind(env) before the episode")
        k = len(self._prop)
        v = self.obs_normalizer.normalize(np.asarray(obs.vector).astype(np.float32))
        self._prop.append(v.astype(np.float32)[self.lo:self.hi])
        if k in (self._next_decision - 1, self._next_decision):
            self._pts[k] = self._cloud()

    def act(self, obs: Any, trace: Any = None) -> np.ndarray:
        """First ``T_a`` normalized actions of the predicted chunk, ``[T_a, A]`` float64."""
        n = len(self._prop)
        if n - 1 != self._next_decision:
            raise RuntimeError(f"decision at frame {n - 1}, expected {self._next_decision}")
        w = window_indices(n, n - 1, self.obs_horizon, 1)
        missing = [int(i) for i in w.obs_idx if int(i) not in self._pts]
        if missing:
            raise RuntimeError(f"frames {missing} have no cloud")
        pts = torch.from_numpy(np.stack([self._pts[int(i)] for i in w.obs_idx]))[None]
        prop = torch.from_numpy(np.stack([self._prop[int(i)] for i in w.obs_idx]))[None]
        valid = torch.from_numpy(w.obs_valid)[None]
        out = self.model.predict(pts, prop, valid, self._gen)[0]
        self.n_decisions += 1
        self._next_decision = n - 1 + self.exec_horizon
        for i in [i for i in self._pts if i < n - 1]:  # keep memory flat
            del self._pts[i]
        return out[: self.exec_horizon].numpy().astype(np.float64)

    def close(self) -> None:
        """Release the GL contexts."""
        if self._maker is not None:
            self._maker.close()
            self._maker = None


def _checkpoint(run_dir: Path, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    man = json.loads((Path(run_dir) / "manifest.json").read_text())
    ck = {c["step"]: c for c in man["checkpoints"]}
    if step not in ck:
        raise KeyError(f"{man['run_id']} has no checkpoint at step {step}: {sorted(ck)}")
    return man, ck[step]


def make_policy(arm: str, run_dir: Path, step: int,
                expected_sha256: str | None = None) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The arm's closed-loop policy for one checkpoint, hash-checked against its manifest (and
    against ``expected_sha256``, the committed selection, when given)."""
    man, ck = _checkpoint(run_dir, step)
    if expected_sha256 is not None and ck["sha256"] != expected_sha256:
        raise ValueError(f"{man['run_id']} step {step}: manifest {ck['sha256']} != "
                         f"expected {expected_sha256}")
    path = Path(run_dir) / ck["path"]
    if arm == "pc":
        return PCClosedLoopPolicy(load_pc_checkpoint(path, ck["sha256"]), ck["sha256"]), man, ck
    if arm == "rgb":
        from roboforge.eval.rgb import RGBClosedLoopPolicy, load_rgb_checkpoint

        return RGBClosedLoopPolicy(load_rgb_checkpoint(path, ck["sha256"]), ck["sha256"]), man, ck
    raise ValueError(f"unknown arm {arm!r}")


def evaluate(arm: str, run_dir: Path, step: int, *, suite: str, out_root: Path,
             expected_sha256: str | None = None, n_states: int | None = None,
             n_verify: int = 3, label: str = "") -> dict[str, Any]:
    """One checkpoint of ``arm`` on a Lift validation-regime ``suite``; returns the summary."""
    from roboforge.eval.rgb import run_rgb_states
    from roboforge.eval.validation import (
        _summarize,
        canonical_platform_id,
        check_suite_allowed,
        eval_run_id,
    )
    from roboforge.paths import repo_relative
    from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
    from roboforge.sim.eval_states import load_suite, read_manifest
    from roboforge.train.run import write_manifest

    policy, man, ck = make_policy(arm, run_dir, step, expected_sha256)
    task = man["task"]
    check_suite_allowed(task, suite)
    states = load_suite(task, suite)
    partial = n_states is not None and n_states < len(states)
    states = states[:n_states] if partial else states
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    eval_id = eval_run_id(man["run_id"], step, platform["platform_id"],
                          stochastic=policy.stochastic, suite=suite,
                          n_partial=len(states) if partial else None)
    out_dir = Path(out_root) / eval_id
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        run = run_rgb_states(policy, states, eval_id, out_dir, task, n_verify)
    finally:
        if hasattr(policy, "close"):
            policy.close()
    sm = read_manifest(task, suite)
    summary = {
        "schema": SCHEMA, "eval_run_id": eval_id, "arm": arm,
        "kind": f"closed-loop {suite} evaluation (Level 2 study){label}",
        "created_utc": datetime.now(tz=UTC).isoformat(), "train_run_id": man["run_id"],
        "family": man["family"], "seed": man["seed"],
        "checkpoint": {"step": step, **ck}, "policy": policy.info.to_dict(),
        "task": task, "suite": suite, "partition": sm["partition"],
        "suite_manifest_content_sha256": sm["content_sha256"],
        "stochastic_policy": ({"policy_seed": 0, "repeat": 0, "samples_per_decision": 1}
                              if policy.stochastic else None),
        "platform": platform, "canonical_platform": canonical,
        "label": ("canonical platform (D-017)" if canonical
                  else "DEVELOPMENT ONLY: native platform, never pooled with canonical results"),
        "eligible_for_selection": bool(canonical and not partial and suite == "validation"),
        "partial": partial, "git": git_revision(), "environment": environment_snapshot(),
        "episode_rules": run["episode_rules"], **_summarize(run["episodes"], run["action_dim"]),
        "renders": run["renders"], "wall_s": run["wall_s"],
        "verify_record": run["verified"],
        "verify_passed": all(v["passed"] for v in run["verified"]),
        "records_dir": repo_relative(out_dir), "episodes": run["episodes"]}
    write_manifest(out_dir / "manifest.json", summary)
    return summary


__all__ = ["SCHEMA", "PCClosedLoopPolicy", "evaluate", "load_pc_checkpoint", "make_policy"]
