"""Non-learned sanity policies and demonstration-replay instrumentation (MASTER_SPEC §5.2, §3.3).

None of these is a learned policy, and their records say so. ``PolicyInfo`` carries machine-
readable flags that every rollout record and export copies:

* ``learned``: trained on demonstrations (always False here);
* ``autonomous``: actions are produced at execution time without recorded human actions.
  Random and no-op policies are autonomous (they do not use the observation either);
  **demonstration replay is not**: it plays recorded teleoperation actions open-loop and must
  never be counted as policy success.

Definitions (DECISIONS D-027):

* ``random_uniform``: i.i.d. uniform actions over the full ActionSpec bounds ([−1, 1]^7), from a
  private ``np.random.Generator`` seeded by ``(policy_seed, state_id)``. A private generator is
  required because restoring a snapshot calls ``env.reset()``, which draws from the global
  ``np.random``.
* ``noop``: the zero action: zero Cartesian position and orientation deltas, and gripper command
  0. robosuite's Panda gripper integrates ``sign(command)``, so 0 holds the gripper target
  where it is.
* ``demo_replay``: the recorded action sequence of one demonstration, one action per step.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from roboforge.data.specs import ActionSpec


@dataclass(frozen=True)
class PolicyInfo:
    """Machine-readable identity of a policy, copied into every rollout record."""

    name: str
    kind: str  # "sanity_baseline" | "instrumentation" | "learned"
    learned: bool
    autonomous: bool
    action_space: str  # "env": clipped env actions | "normalized": decoded by ActionCodec
    label: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return asdict(self)


class Policy(Protocol):
    """Closed-loop interface used by the rollout runner.

    Policies may also define ``observe(obs)``. The runner then calls it with every frame's
    observation, after the reset and after each executed action (D-032). Learned policies
    (:class:`roboforge.eval.learned.LearnedPolicy`) use it to keep their observation history.
    """

    info: PolicyInfo

    def reset_episode(self, episode: Any) -> None:
        """Called once after every environment reset."""

    def act(self, obs: Any) -> np.ndarray:
        """Return an action chunk ``[k, action_dim]`` (k ≥ 1) executed before re-querying."""


def episode_seed(policy_seed: int, state_id: str) -> list[int]:
    """Deterministic per-episode seed material from ``(policy_seed, state_id)``."""
    digest = hashlib.sha256(state_id.encode()).digest()
    return [int(policy_seed), int.from_bytes(digest[:8], "little")]


class RandomPolicy:
    """Uniform random actions over the ActionSpec bounds; ignores observations."""

    def __init__(self, action_spec: ActionSpec, policy_seed: int) -> None:
        self.low = np.array(action_spec.low, dtype=np.float64)
        self.high = np.array(action_spec.high, dtype=np.float64)
        self.policy_seed = int(policy_seed)
        self.info = PolicyInfo(
            name="random_uniform", kind="sanity_baseline", learned=False, autonomous=True,
            action_space="env",
            label="non-learned sanity baseline: i.i.d. uniform actions over the ActionSpec "
                  "bounds (ignores observations)",
            params={"policy_seed": self.policy_seed,
                    "episode_rng": "np.random.default_rng([policy_seed, "
                                   "uint64(sha256(state_id)[:8])])"})
        self._rng: np.random.Generator | None = None

    def reset_episode(self, episode: Any) -> None:
        """Re-seed the private generator for this ``(policy_seed, state_id)``."""
        state_id = episode.initial_state["state_id"]
        self._rng = np.random.default_rng(episode_seed(self.policy_seed, state_id))

    def act(self, obs: Any) -> np.ndarray:
        """One uniform action."""
        if self._rng is None:
            raise RuntimeError("reset_episode() first")
        return self._rng.uniform(self.low, self.high)[None, :]


class NoOpPolicy:
    """The zero action every step (hold pose; gripper target unchanged)."""

    def __init__(self, action_spec: ActionSpec) -> None:
        self.dim = action_spec.dim
        self.info = PolicyInfo(
            name="noop", kind="sanity_baseline", learned=False, autonomous=True,
            action_space="env",
            label="non-learned sanity baseline: zero Cartesian delta and gripper command 0 "
                  "(hold) every step")

    def reset_episode(self, episode: Any) -> None:
        """Stateless."""

    def act(self, obs: Any) -> np.ndarray:
        """The zero action."""
        return np.zeros((1, self.dim))


class DemoReplayPolicy:
    """Open-loop playback of one demonstration's recorded actions (NOT autonomous)."""

    def __init__(self, actions: np.ndarray, demo: str, dataset_sha256: str) -> None:
        self.actions = np.asarray(actions, dtype=np.float64)
        self.info = PolicyInfo(
            name="demo_replay", kind="instrumentation", learned=False, autonomous=False,
            action_space="env",
            label="NON-AUTONOMOUS instrumentation: open-loop playback of recorded human "
                  "teleoperation actions; not a policy result and never counted as success",
            params={"demo": demo, "n_actions": int(len(self.actions)),
                    "dataset_low_dim_sha256": dataset_sha256})
        self._t = 0

    def reset_episode(self, episode: Any) -> None:
        """Restart from the first recorded action."""
        self._t = 0

    def act(self, obs: Any) -> np.ndarray:
        """The next recorded action; ``StopIteration`` when the recording is exhausted."""
        if self._t >= len(self.actions):
            raise StopIteration("demonstration actions exhausted")
        a = self.actions[self._t]
        self._t += 1
        return a[None, :]
