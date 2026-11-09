"""Denoising explorer data: every decision of a recorded validation episode (D-089 rule 4).

Built on the Session 4 truthful-trace method (:mod:`roboforge.viz.denoising`): the frozen
checkpoint re-runs every decision of a **validation** record with the episode's own noise seed,
and each decision's chunk must equal the recorded ``raw_policy_action`` bitwise. For every
decision this module keeps:

* the DDIM steps: scheduler timestep, the sampler state ``A_t`` entering the step, the
  predicted clean chunk ``x̂₀`` (after ``clip_sample``) and the step output, in the normalized
  action space, plus the final chunk decoded to simulator actions by the policy's codec;
* the conditioning window (normalized observations, padding mask) and its observation names;
* 16 **diversity** samples from the same window (seeds ``1_000_000 + i``), never executed;
* clone-simulated end-effector **paths** (:mod:`roboforge.viz.obs_branch`) of each ``A_t``, each
  ``x̂₀``, each diversity sample and the final chunk, next to the recorded executed path.

Every array is a proposal unless labelled recorded; the labels travel with the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.viz.denoising import (
    LABEL_FINAL,
    LABEL_INTERMEDIATE,
    decision_frames,
    decision_trace,
    diversity_samples,
    replay_episode,
)
from roboforge.viz.obs_branch import LABEL_BRANCH, LABEL_EXECUTED

DENOISE_SCHEMA = "roboforge.observatory_denoise/1"
DIVERSITY_N, DIVERSITY_SEED_BASE = 16, 1_000_000
R = 5  # decimals for normalized and simulator actions


def _r(a: Any, d: int = R) -> list:
    return np.round(np.asarray(a, dtype=np.float64), d).tolist()


def episode_traces(policy: Any, record: Any, codec: Any, run_branches: Any,
                   eef_index: int) -> dict[str, Any]:
    """All decisions of one validation record (see the module docstring).

    ``run_branches(jobs)`` executes ``[(frame, actions, want_states)]`` clone simulations and
    returns one ``{"path", "states", "ended"}`` per job, in order
    (:func:`roboforge.viz.obs_branch.run_serial` or a process pool)."""
    frames = decision_frames(record)
    caps = replay_episode(policy, record, {d for d, _, _ in frames})
    body_pos = record.arrays["body_pos"]
    states = record.arrays["state"]
    staged, jobs = [], []
    for d, first, n in frames:
        cap = caps[d]
        meta, arrays = decision_trace(policy, record, cap, codec)
        div = diversity_samples(policy, cap, DIVERSITY_N, DIVERSITY_SEED_BASE)
        f = cap["frame"]
        dec = lambda chunk: codec.decode(np.asarray(chunk))[0]  # noqa: E731
        div_env = [dec(x) for x in div["chunks_normalized"]]
        cands = ([("final", arrays["final_chunk_env_action"])]
                 + [("sample_in", dec(x)) for x in arrays["sample_in_normalized"]]
                 + [("x0", dec(x)) for x in arrays["pred_original_normalized"]]
                 + [("diversity", x) for x in div_env])
        base = len(jobs)
        jobs += [(f, a, kind == "final") for kind, a in cands]
        staged.append((d, first, n, f, meta, arrays, div, div_env, cands, base))
    results = run_branches(jobs)
    decisions = []
    for d, first, n, f, meta, arrays, div, div_env, cands, base in staged:
        res = results[base:base + len(cands)]
        by: dict[str, list] = {"sample_in": [], "x0": [], "diversity": []}
        for (kind, _), r in zip(cands, res, strict=True):
            if kind != "final":
                by[kind].append(_r(r["path"], 6))
        executed = res[0]
        got = executed["states"][:n]
        ok = bool(len(got) == min(n, len(states) - f - 1)
                  and np.array_equal(got, states[f + 1:f + 1 + len(got)]))
        if not ok:
            raise AssertionError(f"decision {d}: executed branch differs from the record")
        decisions.append({
            "decision": d, "frame": f, "first_action_index": first, "n_executed": n,
            "timesteps": [int(x) for x in arrays["timesteps"]],
            "prev_timesteps": [int(x) for x in arrays["prev_timesteps"]],
            "sample_in_normalized": _r(arrays["sample_in_normalized"]),
            "x0_normalized": _r(arrays["pred_original_normalized"]),
            "final_normalized": _r(arrays["final_chunk_normalized"]),
            "final_env_action": _r(arrays["final_chunk_env_action"]),
            "final_clip_mask": arrays["final_chunk_clip_mask"].astype(bool).tolist(),
            "obs_window_normalized": _r(arrays["obs_history_normalized"]),
            "obs_valid": arrays["obs_valid"].astype(bool).tolist(),
            "recorded_executed_action": _r(arrays["recorded_executed_action"]),
            "diversity": {"seeds": div["seeds"], "chunks_normalized":
                          _r(div["chunks_normalized"]), "env_action": _r(div_env),
                          "mean_std_normalized": float(
                              div["std_over_samples_per_step_dim"].mean())},
            "paths": {"recorded_executed": _r(body_pos[f:f + n + 1, eef_index], 6),
                      "final": _r(executed["path"], 6), "final_ended": executed["ended"],
                      **by},
            "checks": {**meta["verification"], "executed_branch_equals_record": ok},
            "noise": meta["noise"]["generator_state_sha256_before_call"],
        })
    last = staged[-1][4] if staged else None
    return {"decisions": decisions,
            "scheduler": None if last is None else last["scheduler"],
            "noise_rule": None if last is None else last["noise"]["rule"],
            "episode_noise_seed": None if last is None else last["noise"]["episode_noise_seed"],
            "labels": {"intermediate": LABEL_INTERMEDIATE, "final": LABEL_FINAL,
                       "branch": LABEL_BRANCH, "executed_branch": LABEL_EXECUTED,
                       "diversity": "proposals from extra seeds at the same condition; never "
                                    "executed",
                       "recorded": "recorded end-effector path of the executed actions"},
            "normalization": None if last is None else last["normalization"]["action"],
            "observation_names": None if last is None else last["observation_names"],
            "n_branches": len(jobs)}
