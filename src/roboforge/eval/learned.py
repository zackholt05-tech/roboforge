"""Closed-loop adapter for trained policies (MASTER_SPEC §5.1; DECISIONS D-027).

:class:`LearnedPolicy` implements the :class:`roboforge.eval.policies.Policy` protocol for
any network of :mod:`roboforge.models`:

* ``observe(obs)`` is called by the rollout runner after the reset and after every
  executed action. Each observation is cast to float32 first, exactly as the processed
  training data was stored, then normalized with the checkpoint's train-only normalizer.
* ``act(obs)`` builds the window with :func:`roboforge.data.windows.window_indices`, the
  same index arithmetic that built every training window, so a decision at frame ``t`` sees
  exactly what the training window for step ``t`` contained. It runs the network's
  ``forward`` and returns the first ``T_a`` normalized actions of the ``T_p`` chunk. The
  runner decodes them with the checkpoint's :class:`ActionCodec`: unnormalize, then clip to
  the ActionSpec bounds.
* ``reset_episode`` clears the history. No observation, hidden state or RNG carries from
  one episode to the next.

**Stochastic networks (diffusion).** ``act`` draws a seeded sample instead of a forward pass.
Each episode gets its own ``torch.Generator``, created in ``reset_episode`` with the seed
:func:`noise_seed` of ``(policy_seed, state_id, repeat)`` (RESEARCH_PROTOCOL §5), and every
decision of the episode consumes that generator in order (initial noise, then any sampler
noise). The global RNG is never read, so an episode's actions depend only on the checkpoint,
the seed triple and the observations, not on which episodes ran before it.
:meth:`LearnedPolicy.episode_info` reports the seed for the rollout record.

Inference uses CPU, ``torch.no_grad`` and a fixed torch thread count (1 by default), so the
same checkpoint gives bitwise-identical actions, and hence identical ``physics_sha256``, on
one platform. The checkpoint must match the evaluation task's committed contract
(observation, action and task spec hashes, and the normalization hash), or loading raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from roboforge.data.normalization import ActionCodec, Normalizer
from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.specs import ActionSpec
from roboforge.data.windows import window_indices
from roboforge.eval.policies import PolicyInfo, episode_seed
from roboforge.models.registry import build_model, count_parameters, is_stochastic
from roboforge.provenance import sha256_file
from roboforge.train.checkpoint import CheckpointError, load_checkpoint

CONTRACT_HASHES = ("observation_spec_sha256", "action_spec_sha256", "task_spec_sha256")


NOISE_SEED_RULE = ("torch.Generator().manual_seed(SeedSequence([policy_seed, "
                   "uint64(sha256(state_id)[:8], little-endian), repeat])"
                   ".generate_state(1, uint64)[0]), one generator per episode")


def noise_seed(policy_seed: int, state_id: str, repeat: int) -> int:
    """Per-episode sampling seed of a stochastic policy (RESEARCH_PROTOCOL §5)."""
    material = [*episode_seed(policy_seed, state_id), int(repeat)]
    return int(np.random.SeedSequence(material).generate_state(1, np.uint64)[0])


def _build(payload: dict[str, Any], state: dict[str, Any]) -> torch.nn.Module:
    d, cfg = payload["dims"], payload["config"]
    model = build_model(payload["family"], cfg["model"], d["obs_dim"], d["action_dim"],
                        d["obs_horizon"], d["pred_horizon"])
    model.load_state_dict(state, strict=True)
    if count_parameters(model) != payload["param_count"]:
        raise CheckpointError("parameter count differs from the checkpoint's record")
    return model.eval()


def trained_model_from_payload(payload: dict[str, Any]) -> torch.nn.Module:
    """The trained (non-EMA) network of a checkpoint, loaded strictly, in eval mode."""
    return _build(payload, payload["model_state"])


def model_from_payload(payload: dict[str, Any]) -> torch.nn.Module:
    """The **inference** network of a checkpoint (strict, eval mode): the EMA weights when
    the run used EMA (diffusion), otherwise the trained weights."""
    if payload["config"]["model"].get("ema"):
        if "ema_state" not in payload:
            raise CheckpointError("config enables EMA but the checkpoint has no ema_state")
        return _build(payload, payload["ema_state"]["model_state"])
    return trained_model_from_payload(payload)


def check_committed_contract(task: str, hashes: dict[str, str]) -> None:
    """Raise unless ``hashes`` match the task's committed specs and normalization."""
    docs = load_task_artifacts(task)
    m = docs["task_manifest"]
    want = {k: m[k] for k in CONTRACT_HASHES}
    want["normalization_sha256"] = docs["normalization"]["content_sha256"]
    bad = {k: (hashes.get(k), v) for k, v in want.items() if hashes.get(k) != v}
    if bad:
        raise CheckpointError(f"checkpoint does not match the committed {task} contract: {bad}")


class LearnedPolicy:
    """A trained network as a closed-loop policy in the normalized action space."""

    def __init__(self, model: torch.nn.Module, meta: dict[str, Any], *, checkpoint_sha256: str,
                 checkpoint_path: str | None = None, num_threads: int = 1,
                 policy_seed: int = 0, repeat: int = 0) -> None:
        torch.set_num_threads(int(num_threads))
        self.model = model.eval()
        self.stochastic = is_stochastic(self.model)
        self.policy_seed, self.repeat = int(policy_seed), int(repeat)
        self.task = meta["task"]
        self.hashes = dict(meta["hashes"])
        check_committed_contract(self.task, self.hashes)
        d = meta["dims"]
        self.obs_horizon, self.pred_horizon = int(d["obs_horizon"]), int(d["pred_horizon"])
        self.exec_horizon = int(d["exec_horizon"])
        self.obs_normalizer = Normalizer.from_dict(meta["normalization"]["obs"])
        self.action_normalizer = Normalizer.from_dict(meta["normalization"]["action"])
        name = meta["config"]["name"]
        self.info = PolicyInfo(
            name=name, kind="learned", learned=True, autonomous=True,
            action_space="normalized",
            label=(f"learned {meta['family']} policy '{name}' trained on robomimic PH "
                   f"demonstrations (privileged low-dim state); checkpoint "
                   f"{checkpoint_sha256[:12]} at step {meta['global_step']}"),
            params={"run_id": meta["run_id"], "family": meta["family"],
                    "global_step": int(meta["global_step"]),
                    "checkpoint_sha256": checkpoint_sha256, "checkpoint_path": checkpoint_path,
                    "config_sha256": meta["config_sha256"], "param_count": meta["param_count"],
                    "horizons": {"obs": self.obs_horizon, "pred": self.pred_horizon,
                                 "exec": self.exec_horizon},
                    "num_threads": torch.get_num_threads(), "device": "cpu",
                    "torch_version": str(torch.__version__)})
        if self.stochastic:
            self.info.params.update(
                policy_seed=self.policy_seed, repeat=self.repeat, noise_seed_rule=NOISE_SEED_RULE,
                inference_network="ema" if meta["config"]["model"]["ema"] else "trained",
                diffusion=self.model.describe())
        self._hist: list[np.ndarray] = []
        self.n_decisions = 0
        self._gen: torch.Generator | None = None
        self._episode: dict[str, Any] | None = None

    @classmethod
    def from_checkpoint(cls, path: Any, expected_sha256: str | None = None,
                        num_threads: int = 1, policy_seed: int = 0,
                        repeat: int = 0) -> LearnedPolicy:
        """Load a checkpoint file (hash-verified when ``expected_sha256`` is given).
        ``policy_seed`` and ``repeat`` only affect stochastic networks."""
        sha = sha256_file(path)
        if expected_sha256 is not None and sha != expected_sha256:
            raise CheckpointError(f"{path}: SHA-256 {sha} != expected {expected_sha256}")
        payload = load_checkpoint(path, expected_sha256=sha)
        return cls(model_from_payload(payload), payload, checkpoint_sha256=sha,
                   checkpoint_path=str(path), num_threads=num_threads,
                   policy_seed=policy_seed, repeat=repeat)

    def codec(self, action_spec: ActionSpec) -> ActionCodec:
        """Decoder from normalized model actions to clipped simulator actions."""
        return ActionCodec(self.action_normalizer, np.array(action_spec.low, dtype=np.float64),
                           np.array(action_spec.high, dtype=np.float64))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector).astype(np.float32)  # training data are stored as float32
        return self.obs_normalizer.normalize(v).astype(np.float32)

    def reset_episode(self, episode: Any) -> None:
        """Clear the observation history; refuse an episode of another task or contract."""
        if episode.task != self.task:
            raise ValueError(f"policy trained on {self.task!r}, episode is {episode.task!r}")
        for k in CONTRACT_HASHES:
            if getattr(episode, k) != self.hashes[k]:
                raise CheckpointError(f"episode {k} differs from the checkpoint's")
        self._hist = []
        self.n_decisions = 0
        if self.stochastic:
            state_id = episode.initial_state["state_id"]
            seed = noise_seed(self.policy_seed, state_id, self.repeat)
            self._gen = torch.Generator().manual_seed(seed)
            self._episode = {"state_id": state_id, "policy_seed": self.policy_seed,
                             "repeat": self.repeat, "noise_seed": seed,
                             "noise_seed_rule": NOISE_SEED_RULE}

    def episode_info(self) -> dict[str, Any] | None:
        """Per-episode sampling provenance (stochastic policies), stored in the record."""
        return None if self._episode is None else dict(self._episode)

    def observe(self, obs: Any) -> None:
        """Record one frame's observation (called after reset and after every step)."""
        self._hist.append(self._normalize(obs.vector))

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """``(obs_history [T_o, D], obs_valid [T_o])`` for a decision at the newest frame."""
        n = len(self._hist)
        w = window_indices(n, n - 1, self.obs_horizon, 1)
        return np.stack([self._hist[i] for i in w.obs_idx]), w.obs_valid

    def act(self, obs: Any, trace: Any = None) -> np.ndarray:
        """First ``T_a`` normalized actions of the predicted chunk, ``[T_a, A]`` float64.

        ``trace`` (stochastic policies only) receives every sampler step of this decision's
        call; the denoising export uses it, the rollout runner never does."""
        if not self._hist:
            raise RuntimeError("observe() must be called before act() (rollout runner hook)")
        if not np.array_equal(self._normalize(obs.vector), self._hist[-1]):
            raise RuntimeError("act() observation differs from the last observed frame")
        oh, ov = self.window()
        x, v = torch.from_numpy(oh)[None], torch.from_numpy(ov)[None]
        with torch.no_grad():
            if self.stochastic:
                if self._gen is None:
                    raise RuntimeError("reset_episode() must seed the sampler first")
                out = self.model.sample(x, v, self._gen, trace=trace)[0]
            else:
                out = self.model(x, v)[0]
        self.n_decisions += 1
        return out[: self.exec_horizon].numpy().astype(np.float64)
