"""Serialized initial simulator snapshots and their exact restoration (DECISIONS D-025).

A snapshot is the pair (MJCF string, flattened MuJoCo state ``[time, qpos, qvel]``) plus
labels: task, a stable ``state_id``, its partition and where it came from.

**Which XML.** The stored string is MuJoCo's own serialization of the *live* model
(``env.sim.model.get_xml()``, i.e. ``mj_saveLastXML``), the same convention as the robomimic
datasets' ``model_file``. robosuite's generator XML (``env.model.get_xml()``) is **not** the
model that ran: Lift writes a freshly sampled cube size into it on every reset, and
PickPlace edits ``body_pos``/``body_quat`` of the bins and target-bin visual objects at
runtime, after compilation (Session 2 probe: up to 0.9 m). The canonical evaluation model is
``compile(env.edit_model_xml(stored_string))``. Every process compiles exactly that string;
nothing is ever re-serialized from a loaded model (KI-016).

**Portability.** MuJoCo writes absolute asset paths (``/Users/.../site-packages/robosuite/
models/assets/...``). :func:`portable_xml` rewrites every ``file="..."`` attribute to the
robosuite-relative form ``robosuite/models/assets/...``, which ``edit_model_xml`` rebases
onto the local robosuite install. Nothing else in the string changes; :func:`capture`
verifies that ``edit_model_xml`` of the portable and of the original string are identical.
Hashes are therefore platform-independent.

**Hidden state.** Some simulator state is not in the flattened vector: MuJoCo's
``qacc_warmstart``, ``ctrl`` and mocap poses, robosuite's gripper ``current_action``
integrator, controller goals and the step counter. :func:`restore` sets all of them to their
reset values by compiling a fresh ``MjSim`` and running robosuite's deterministic reset
before writing the state (``roboforge.sim.state_replay.reset_to_state``). A snapshot is
therefore only meaningful as an *episode start*.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roboforge.sim.state_replay import reset_to_state

SNAPSHOT_SCHEMA = "roboforge.initial_state/1"
# Demonstration episodes (offline data only), the two simulator evaluation suites, the
# Session 7 robustness starts (validation regime, never selection; D-063), and ad-hoc
# development starts (seeds outside every reserved block; never reported).
PARTITIONS = ("demo_train", "demo_valid", "eval_validation", "eval_final_test", "eval_robustness",
              "eval_adhoc")
_FILE_ATTR = re.compile(r'(\bfile=")([^"]*)(")')
_ABS_ATTR = re.compile(r'="(/[^"]*)"')


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_state(state: np.ndarray) -> str:
    """SHA-256 of a flattened state as little-endian float64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(state, dtype="<f8").tobytes()).hexdigest()


def portable_xml(xml: str) -> str:
    """Rewrite absolute robosuite asset paths to ``robosuite/<path inside the package>``.

    Raises ``ValueError`` if an asset path has no ``robosuite`` directory component, or if
    any absolute path remains afterwards (the result must not depend on the machine).
    """

    def repl(m: re.Match[str]) -> str:
        parts = m.group(2).split("/")
        hits = [i for i, p in enumerate(parts) if p == "robosuite"]
        if not hits:
            raise ValueError(f"asset path without a 'robosuite' component: {m.group(2)!r}")
        return m.group(1) + "/".join(["robosuite", *parts[max(hits) + 1:]]) + m.group(3)

    out = _FILE_ATTR.sub(repl, xml)
    leftover = _ABS_ATTR.findall(out)
    if leftover:
        raise ValueError(f"absolute paths remain after normalization: {leftover[:3]}")
    return out


@dataclass(frozen=True)
class InitialState:
    """One serialized episode start: portable MJCF string + flattened MuJoCo state."""

    task: str
    state_id: str
    partition: str
    model_xml: str
    state: np.ndarray
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition {self.partition!r} not in {PARTITIONS}")
        s = np.asarray(self.state, dtype=np.float64)
        if s.ndim != 1 or not np.all(np.isfinite(s)):
            raise ValueError("state must be a finite 1-D float64 vector")
        object.__setattr__(self, "state", s)

    @property
    def model_xml_sha256(self) -> str:
        """Hash of the stored (portable) XML string."""
        return sha256_text(self.model_xml)

    @property
    def state_sha256(self) -> str:
        """Hash of the flattened state."""
        return sha256_state(self.state)

    def reference(self) -> dict[str, Any]:
        """Labels and hashes (no XML, no state values): what run manifests record."""
        return {"schema": SNAPSHOT_SCHEMA, "task": self.task, "state_id": self.state_id,
                "partition": self.partition, "model_xml_sha256": self.model_xml_sha256,
                "state_sha256": self.state_sha256, "state_dim": int(self.state.shape[0]),
                "source": dict(self.source)}

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-compatible form (floats round-trip exactly through ``repr``)."""
        return {**self.reference(), "model_xml": self.model_xml, "state": self.state.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InitialState:
        """Rebuild from :meth:`to_dict` output, verifying both hashes."""
        init = cls(task=d["task"], state_id=d["state_id"], partition=d["partition"],
                   model_xml=d["model_xml"], state=np.asarray(d["state"], dtype=np.float64),
                   source=dict(d.get("source", {})))
        if init.model_xml_sha256 != d["model_xml_sha256"]:
            raise ValueError(f"{init.state_id}: model XML hash mismatch")
        if init.state_sha256 != d["state_sha256"]:
            raise ValueError(f"{init.state_id}: state hash mismatch")
        return init


def capture(env: Any, task: str, state_id: str, partition: str,
            source: dict[str, Any]) -> InitialState:
    """Snapshot the current live model and state of a robosuite ``env`` (just after reset).

    Verifies that the portable XML compiles to exactly the same string as the original under
    ``env.edit_model_xml``, so normalization changed nothing but asset paths.
    """
    raw = env.sim.model.get_xml()
    xml = portable_xml(raw)
    if env.edit_model_xml(xml) != env.edit_model_xml(raw):
        raise AssertionError(f"{state_id}: path normalization changed the compiled XML")
    return InitialState(task=task, state_id=state_id, partition=partition, model_xml=xml,
                        state=np.array(env.sim.get_state().flatten()), source=source)


def generate(env: Any, task: str, seed: int, state_id: str, partition: str,
             extra_source: dict[str, Any] | None = None) -> InitialState:
    """Draw one episode start from robosuite's standard reset distribution.

    robosuite v1.5.1 samples placement, Lift's cube size and robot joint noise from the
    global ``np.random`` (KI-002); Python's ``random`` is seeded too, in case any code path
    uses it. The start is a function of ``seed`` only: a hard reset rebuilds the model.
    """
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    source = {"kind": "generated", "method": "seed np.random and random, then env.reset()",
              "seed": int(seed), **(extra_source or {})}
    return capture(env, task, state_id, partition, source)


def restore(env: Any, init: InitialState) -> np.ndarray:
    """Hard-reset ``env`` to ``init`` (fresh MjSim + deterministic reset); return read-back.

    Raises ``AssertionError`` unless the read-back flattened state equals ``init.state``
    exactly.
    """
    achieved = reset_to_state(env, init.model_xml, init.state)
    if achieved.shape != init.state.shape or not np.array_equal(achieved, init.state):
        raise AssertionError(f"{init.state_id}: restored state differs from the snapshot")
    return achieved


def from_demo(task: str, demo: str, model_file: str, states: np.ndarray, partition: str,
              dataset_sha256: str) -> InitialState:
    """Episode start of a recorded demonstration (its own dataset XML and ``states[0]``)."""
    if partition not in ("demo_train", "demo_valid"):
        raise ValueError(f"demonstration partition must be demo_train/demo_valid: {partition}")
    return InitialState(task=task, state_id=f"{task}_{demo}", partition=partition,
                        model_xml=portable_xml(model_file), state=np.asarray(states[0]),
                        source={"kind": "demonstration", "demo": demo,
                                "dataset_low_dim_sha256": dataset_sha256})
