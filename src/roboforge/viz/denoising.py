"""Truthful denoising traces of real sampler calls (MASTER_SPEC §5.5, §8.4; Session 4).

:func:`replay_episode` re-runs a trained diffusion policy over the **recorded observations**
of one closed-loop validation episode, with the episode's own noise seed
(``(policy_seed, state_id, repeat)``, RESEARCH_PROTOCOL §5). Every decision therefore
consumes the policy's generator exactly as it did during the rollout. Each decision's
sampled chunk is checked **bitwise** against the actions the rollout recorded
(``raw_policy_action``); the export refuses to proceed otherwise. So every exported
intermediate tensor comes from the same sampler call that produced the executed actions. The
check can only pass on the platform that recorded the episode (KI-013), so export on the
canonical container for canonical episodes.

For the selected decisions, :func:`decision_trace` stores, per reverse step ``k``:

* the scheduler timestep ``t_k`` and the next timestep, ``abar_t``, ``abar_prev`` and every
  coefficient the step used;
* ``A_{t_k}`` (the step input), the predicted noise, ``x0_hat`` before and after
  ``clip_sample``, the injected noise (``None`` for DDIM with ``eta = 0``) and the step
  output, all in the **normalized** action space;
* the same chunks **denormalized** with the train-only action normalizer (not clipped).

It also stores the conditioning window (raw float64 recorded observations, the float32 cast
the policy normalizes, the normalized window and its padding mask), the episode noise seed,
the decision index and the final chunk decoded to simulator actions.

**Labels (binding for any consumer, e.g. the Observatory).** Intermediate states and the
final chunk are *proposals*: normalized action sequences inside the sampler, not physical
paths. Only the first ``n_executed`` actions of the final chunk were executed. The arrays are
never converted to world-space trajectories here.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from roboforge.eval.learned import NOISE_SEED_RULE, LearnedPolicy, noise_seed
from roboforge.eval.rollout import RolloutRecord
from roboforge.models.noise_schedule import StepRecord

SCHEMA = "roboforge.denoising_trace/1"
LABEL_INTERMEDIATE = ("proposal: intermediate state of the reverse diffusion sampler in "
                      "normalized action space; not a physical path and never executed")
LABEL_FINAL = ("proposal: final sampled action chunk; only the first n_executed actions were "
               "sent to the simulator, the rest were discarded by receding-horizon control")


def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}".encode() + a.tobytes()).hexdigest()


def decision_frames(record: RolloutRecord) -> list[tuple[int, int, int]]:
    """``(decision, first action index, n actions)`` of every decision in the record."""
    di = record.arrays["decision_index"]
    out = []
    for d in np.unique(di):
        idx = np.flatnonzero(di == d)
        out.append((int(d), int(idx[0]), int(len(idx))))
    return out


def replay_episode(policy: LearnedPolicy, record: RolloutRecord,
                   capture: set[int]) -> dict[int, dict[str, Any]]:
    """Re-run every decision of ``record``; return the captured sampler calls by decision.

    Raises ``AssertionError`` if any decision's chunk differs from the recorded actions.
    """
    if not policy.stochastic:
        raise ValueError("denoising traces need a stochastic (diffusion) policy")
    meta = record.meta
    if meta["policy"]["params"]["checkpoint_sha256"] != policy.info.params["checkpoint_sha256"]:
        raise ValueError("record was produced by a different checkpoint")
    ep = meta["policy_episode"]
    if (ep["policy_seed"], ep["repeat"]) != (policy.policy_seed, policy.repeat):
        raise ValueError(f"record used seeds {ep}, policy has ({policy.policy_seed}, "
                         f"{policy.repeat})")
    episode = SimpleNamespace(task=meta["task"], initial_state=meta["episode"]["initial_state"],
                              **{k: meta["episode"][k] for k in (
                                  "observation_spec_sha256", "action_spec_sha256",
                                  "task_spec_sha256")})
    policy.reset_episode(episode)
    if policy.episode_info()["noise_seed"] != ep["noise_seed"]:
        raise AssertionError("noise seed rule does not reproduce the recorded seed")
    obs = record.arrays["obs"]
    raw = record.arrays["raw_policy_action"]
    frames = decision_frames(record)
    captured: dict[int, dict[str, Any]] = {}
    frame = 0
    policy.observe(SimpleNamespace(vector=obs[0]))
    for d, first, n in frames:
        while frame < first:  # frames produced by the previous decision's actions
            frame += 1
            policy.observe(SimpleNamespace(vector=obs[frame]))
        steps: list[StepRecord] = []
        oh, ov = policy.window()
        gen_state = policy._gen.get_state().numpy().copy()
        chunk = policy.act(SimpleNamespace(vector=obs[frame]), trace=steps.append)
        if not np.array_equal(chunk[:n], raw[first:first + n]):
            raise AssertionError(f"decision {d}: replayed chunk differs from the recorded "
                                 f"raw_policy_action (max |diff| "
                                 f"{np.abs(chunk[:n] - raw[first:first + n]).max():.3e})")
        if d in capture:
            captured[d] = {"decision": d, "frame": frame, "first_action_index": first,
                           "n_executed": n, "obs_history_normalized": oh, "obs_valid": ov,
                           "generator_state_sha256": hashlib.sha256(gen_state.tobytes())
                           .hexdigest(), "steps": steps,
                           "chunk_first_t_a": chunk}
    return captured


def decision_trace(policy: LearnedPolicy, record: RolloutRecord, cap: dict[str, Any],
                   codec: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Metadata and arrays of one captured decision (see the module docstring)."""
    steps: list[StepRecord] = cap["steps"]
    an = policy.action_normalizer

    def stack(get) -> np.ndarray:
        return np.stack([get(s)[0].numpy().astype(np.float64) for s in steps])

    arrays = {
        "sample_in_normalized": stack(lambda s: s.sample),
        "model_output_eps": stack(lambda s: s.model_output),
        "pred_original_unclipped_normalized": stack(lambda s: s.pred_original_unclipped),
        "pred_original_normalized": stack(lambda s: s.pred_original),
        "prev_sample_normalized": stack(lambda s: s.prev_sample),
        "timesteps": np.array([s.timestep for s in steps], dtype=np.int64),
        "prev_timesteps": np.array([s.prev_timestep for s in steps], dtype=np.int64),
        "obs_history_normalized": cap["obs_history_normalized"].astype(np.float64),
        "obs_valid": cap["obs_valid"].astype(bool)}
    arrays["noise_injected"] = np.array([s.noise is not None for s in steps])
    if arrays["noise_injected"].any():  # DDPM; zeros where a step injected none
        arrays["injected_noise"] = np.stack([
            (s.noise[0].numpy() if s.noise is not None else np.zeros_like(s.sample[0].numpy()))
            .astype(np.float64) for s in steps])
    for k in ("sample_in", "pred_original", "prev_sample"):
        arrays[f"{k}_denormalized"] = an.unnormalize(arrays[f"{k}_normalized"])
    final = arrays["prev_sample_normalized"][-1]
    env_actions, clip = codec.decode(final)
    arrays.update(final_chunk_normalized=final, final_chunk_denormalized=an.unnormalize(final),
                  final_chunk_env_action=env_actions, final_chunk_clip_mask=clip)
    t_o = policy.obs_horizon
    f = cap["frame"]
    window_frames = [max(0, f - t_o + 1 + k) for k in range(t_o)]
    arrays["obs_history_recorded"] = record.arrays["obs"][window_frames]
    arrays["obs_history_float32"] = arrays["obs_history_recorded"].astype(np.float32)
    n = cap["n_executed"]
    first = cap["first_action_index"]
    arrays["recorded_raw_policy_action"] = record.arrays["raw_policy_action"][first:first + n]
    arrays["recorded_executed_action"] = record.arrays["executed_action"][first:first + n]
    ep = record.meta["policy_episode"]
    sampler = policy.model.sampler
    meta = {
        "schema": SCHEMA, "kind": "real sampler call (replayed bitwise from a recorded rollout)",
        "labels": {"intermediate": LABEL_INTERMEDIATE, "final": LABEL_FINAL},
        "eval_run_id": record.meta["run_id"], "task": record.meta["task"],
        "state_id": ep["state_id"], "partition": record.meta["episode"]["initial_state"][
            "partition"], "outcome": record.meta["outcome"],
        "policy": record.meta["policy"], "platform_recorded": record.meta["platform"],
        "decision": cap["decision"], "frame": f, "window_frames": window_frames,
        "first_action_index": first, "n_executed": n,
        "noise": {"episode_noise_seed": ep["noise_seed"], "policy_seed": ep["policy_seed"],
                  "repeat": ep["repeat"], "rule": NOISE_SEED_RULE,
                  "seed_check": noise_seed(ep["policy_seed"], ep["state_id"], ep["repeat"]),
                  "generator_state_sha256_before_call": cap["generator_state_sha256"],
                  "note": "the generator is consumed in order by every decision of the "
                          "episode; this decision's draws follow those of decisions "
                          "0..decision-1"},
        "scheduler": {**policy.model.describe(),
                      "steps": [{"timestep": s.timestep, "prev_timestep": s.prev_timestep,
                                 "alpha_prod_t": s.alpha_prod_t,
                                 "alpha_prod_t_prev": s.alpha_prod_t_prev,
                                 "coefficients": s.coefficients} for s in steps],
                      "alphas_cumprod": policy.model.schedule.alphas_cumprod.tolist()},
        "normalization": {"action": policy.action_normalizer.to_dict(),
                          "normalization_sha256": policy.hashes["normalization_sha256"],
                          "denormalized_means": "unnormalized with the train-only action "
                                                "normalizer, not clipped"},
        "observation_names": record.meta["observation_names"],
        "verification": {"replayed_chunk_equals_recorded_raw_policy_action": True,
                         "final_trace_sample_equals_policy_chunk": bool(np.array_equal(
                             final[:n], cap["chunk_first_t_a"][:n])),
                         "trace_is_contiguous": all(
                             torch.equal(a.prev_sample, b.sample)
                             for a, b in zip(steps, steps[1:], strict=False))},
        "n_steps": len(steps), "sampler": sampler.to_dict(),
        "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str, "sha256": _sha(v)}
                   for k, v in sorted(arrays.items())}}
    return meta, arrays


def diversity_samples(policy: LearnedPolicy, cap: dict[str, Any], n: int,
                      seed_base: int) -> dict[str, Any]:
    """``n`` extra samples from the same conditioning window with seeds ``seed_base + i``.

    These are additional proposals that were **never executed**; they show the spread of the
    policy's action distribution at this condition (MASTER_SPEC §8.4)."""
    x = torch.from_numpy(cap["obs_history_normalized"])[None]
    v = torch.from_numpy(cap["obs_valid"])[None]
    chunks = np.stack([policy.model.sample(x, v, torch.Generator().manual_seed(seed_base + i))[0]
                       .numpy().astype(np.float64) for i in range(n)])
    return {"seeds": [seed_base + i for i in range(n)], "chunks_normalized": chunks,
            "std_over_samples_per_step_dim": chunks.std(axis=0),
            "label": "proposals from extra seeds at the same condition; never executed"}
