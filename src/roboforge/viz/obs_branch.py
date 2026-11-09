"""Candidate end-effector paths by clone simulation (MASTER_SPEC §8.4; DECISIONS D-089).

A diffusion policy's proposals are action chunks, not paths. To show what a proposal *would*
do physically, each candidate chunk is executed in a **clone** of the recorded episode:

1. restore the episode's start (:func:`roboforge.sim.snapshot.restore` through
   ``RoboForgeEnv.reset``, which also resets robosuite's hidden controller and gripper state);
2. re-execute the record's ``executed_action`` up to the decision frame ``f``, checking every
   state **bitwise** against the record, so the clone sits exactly in the recorded state with
   the recorded hidden state;
3. execute the candidate's actions (already decoded and clipped by the policy's codec) and
   record the world position of ``gripper0_right_eef`` after each one.

The branch of the chunk that was actually executed must reproduce the recorded frames
``f+1 .. f+n_executed`` bitwise; :func:`branch` reports it. A branch stops early if the
simulator ends the episode (success or horizon); its length says so.

**Validation records only.** :func:`check_partition` refuses any other partition, so no
held-out start is ever stepped (D-089).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.snapshot import InitialState

BRANCH_PARTITIONS = ("eval_validation",)
LABEL_BRANCH = ("clone simulation of a never-executed proposal: the recorded episode is "
                "re-executed bitwise to the decision frame, then this candidate chunk is run "
                "in the clone; the robot never executed it")
LABEL_EXECUTED = ("clone simulation of the executed chunk; its first n_executed steps "
                  "reproduce the recorded frames bitwise")


def check_partition(initial_state: InitialState) -> None:
    """Refuse to step anything but a validation start."""
    if initial_state.partition not in BRANCH_PARTITIONS:
        raise PermissionError(f"clone simulation is allowed on {BRANCH_PARTITIONS} only, not "
                              f"{initial_state.partition} ({initial_state.state_id})")


class Brancher:
    """Clone simulations of one recorded validation episode."""

    def __init__(self, env: RoboForgeEnv, initial_state: InitialState,
                 executed: np.ndarray, states: np.ndarray, eef_body: str) -> None:
        check_partition(initial_state)
        self.env, self.init = env, initial_state
        self.executed, self.states = np.asarray(executed), np.asarray(states)
        self._eef = eef_body
        self.n_replayed_steps = 0

    def _eef_pos(self) -> np.ndarray:
        kin = self.env.kinematics
        i = kin.body_names.index(self._eef)
        return kin.frame(self.env.current_state())["body_pos"][i]

    def replay_to(self, frame: int) -> None:
        """Reset and re-execute the recorded actions up to ``frame``, bitwise-checked."""
        self.env.reset(self.init)
        if not np.array_equal(self.env.current_state(), self.states[0]):
            raise AssertionError("restored state differs from recorded frame 0")
        for t in range(frame):
            self.env.step(self.executed[t])
            self.n_replayed_steps += 1
            if not np.array_equal(self.env.current_state(), self.states[t + 1]):
                raise AssertionError(f"replay diverged from the record at frame {t + 1}")

    def branch(self, frame: int, actions: np.ndarray) -> dict[str, Any]:
        """EEF positions ``[1 + n, 3]`` after running ``actions`` from recorded ``frame``."""
        self.replay_to(frame)
        path, states, ended = [self._eef_pos()], [], None
        for j, a in enumerate(np.asarray(actions)):
            _, _, term, trunc, info = self.env.step(a)
            self.n_replayed_steps += 1
            path.append(self._eef_pos())
            states.append(self.env.current_state())
            if term or trunc:
                ended = {"after_action": j, "terminated": bool(term), "truncated": bool(trunc),
                         "success": bool(info["success"])}
                break
        return {"path": np.asarray(path), "states": np.asarray(states), "ended": ended}



def run_serial(brancher: Brancher, jobs: list[tuple[int, np.ndarray, bool]]
               ) -> list[dict[str, Any]]:
    """Run clone simulations one after another in this process."""
    out = []
    for frame, actions, want_states in jobs:
        r = brancher.branch(frame, actions)
        out.append({**r, "states": r["states"] if want_states else None})
    return out


_WORKER: dict[str, Any] = {}


def _init_worker(task: str, horizon: int, hold_steps: int, init: dict[str, Any],
                 executed: np.ndarray, states: np.ndarray, eef_body: str) -> None:
    from roboforge.sim.backend import configure_mujoco_gl

    configure_mujoco_gl()
    env = RoboForgeEnv(task, horizon=horizon, hold_steps=hold_steps)
    _WORKER["b"] = Brancher(env, InitialState.from_dict(init), executed, states, eef_body)


def _work(job: tuple[int, np.ndarray, bool]) -> dict[str, Any]:
    return run_serial(_WORKER["b"], [job])[0]


def run_pool(task: str, horizon: int, hold_steps: int, init: InitialState,
             executed: np.ndarray, states: np.ndarray, eef_body: str, workers: int,
             jobs: list[tuple[int, np.ndarray, bool]]) -> list[dict[str, Any]]:
    """Run clone simulations in ``workers`` spawned processes (each its own simulator)."""
    import multiprocessing as mp

    check_partition(init)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(task, horizon, hold_steps, init.to_dict(), executed, states,
                            eef_body)) as pool:
        return pool.map(_work, jobs, chunksize=1)
