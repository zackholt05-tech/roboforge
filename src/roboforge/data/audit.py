"""Dataset audit summaries for ``reports/data_audit.json`` (MASTER_SPEC Session 1).

Every number here is computed from the hash-verified source files and the freshly built
pipeline artifacts. Nothing is typed in by hand. The audit covers episode counts and
durations; saved-time spacing; action ranges and saturation; per-feature state
distributions by split, and how far validation data falls outside the training range; the
success-reward pattern; quaternion sign behaviour; operator metadata; window padding; the
KI-003 correction; and how far robosuite's own closed-loop observations drift from the
canonical features.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from roboforge.data.features import FeatureExtractor, robosuite_transforms
from roboforge.data.inspect import demo_keys
from roboforge.data.normalization import Normalizer
from roboforge.data.processed import stored_pose_blocks
from roboforge.data.sanity import success_tail
from roboforge.data.specs import ObservationSpec


def _dim_stats(x: np.ndarray) -> dict[str, list[float]]:
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist()}


def _summary(v: np.ndarray) -> dict[str, float]:
    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "median": float(np.median(v)), "std": float(v.std())}


def _stack(episodes: dict[str, dict[str, np.ndarray]], demos: list[str], key: str) -> np.ndarray:
    return np.concatenate([episodes[d][key] for d in demos]).astype(np.float64)


def episode_audit(lengths: dict[str, int], split: dict[str, Any], dt: float) -> dict[str, Any]:
    """Episode counts, lengths and durations, overall and per split."""
    out: dict[str, Any] = {"control_dt_s": dt}
    groups = {"all": list(lengths), **split["splits"]}
    for name, ids in groups.items():
        n = np.array([lengths[d] for d in ids])
        out[name] = {"n_episodes": len(ids), "n_steps": int(n.sum()), "length_steps": _summary(n),
                     "duration_s": _summary(n * dt)}
    order = sorted(lengths, key=lambda d: (lengths[d], int(d.split("_")[1])))
    out["shortest"] = {"id": order[0], "length": lengths[order[0]]}
    out["longest"] = {"id": order[-1], "length": lengths[order[-1]]}
    edges = np.arange(0, max(lengths.values()) + 10, 10)
    out["length_histogram"] = {"bin_edges": edges.tolist(), **{
        s: np.histogram([lengths[d] for d in ids], edges)[0].tolist()
        for s, ids in split["splits"].items()}}
    return out


def time_and_success_audit(low: h5py.File) -> dict[str, Any]:
    """Saved-time spacing and the reward/done success pattern at the end of each demo."""
    dts, t0s, tails = [], [], []
    for d in demo_keys(low):
        g = low["data"][d]
        t = g["states"][()][:, 0]
        t0s.append(float(t[0]))
        dts.extend(np.diff(t).tolist())
        tails.append(success_tail(g))
        if not np.array_equal(g["rewards"][()] > 0, g["dones"][()] > 0):
            raise AssertionError(f"{d}: rewards and dones disagree")
    after = np.array([x["steps_after_first_success"] for x in tails])
    return {"state_time_step_s": {"min": min(dts), "max": max(dts)},
            "initial_time_s": {"min": min(t0s), "max": max(t0s)},
            "success_signal": {
                "definition": "reward == done == 1 (sparse reward, robosuite _check_success)",
                "every_demo_ends_successful": all(x["first_success_step"] is not None
                                                  for x in tails),
                "success_persists_to_end_in_all_demos": all(x["persists_to_end"] for x in tails),
                "steps_recorded_after_first_success": _summary(after),
                "rewards_equal_dones": True}}


def action_audit(episodes: dict[str, dict[str, np.ndarray]], split: dict[str, Any],
                 names: list[str]) -> dict[str, Any]:
    """Per-dimension action statistics, gripper values and saturation, per split."""
    out: dict[str, Any] = {"names": names}
    for s, ids in split["splits"].items():
        a = _stack(episodes, ids, "actions")
        out[s] = {**_dim_stats(a),
                  "fraction_abs_ge_0.999": (np.abs(a) >= 0.999).mean(0).tolist(),
                  "gripper_unique_values": sorted(set(a[:, -1].round(6).tolist())),
                  "gripper_fraction_close": float((a[:, -1] > 0).mean())}
    return out


def feature_audit(episodes: dict[str, dict[str, np.ndarray]], split: dict[str, Any],
                  spec: ObservationSpec, obs_norm: Normalizer) -> dict[str, Any]:
    """Per-feature state distributions by split, and validation extrapolation of train ranges."""
    tr, va = _stack(episodes, split["splits"]["train"], "obs"), \
        _stack(episodes, split["splits"]["valid"], "obs")
    lo, hi = tr.min(0), tr.max(0)
    outside = ((va < lo) | (va > hi)).mean(0)
    vn = obs_norm.normalize(va)
    out: dict[str, Any] = {}
    for f, (a, b) in zip(spec.features, spec.slices().values(), strict=True):
        out[f.name] = {"slice": [a, b], "train": _dim_stats(tr[:, a:b]),
                       "valid": _dim_stats(va[:, a:b]),
                       "valid_fraction_outside_train_range": outside[a:b].tolist(),
                       "valid_normalized_min": vn[:, a:b].min(0).tolist(),
                       "valid_normalized_max": vn[:, a:b].max(0).tolist()}
    ignored = [i for i, r in enumerate(hi - lo) if r < obs_norm.range_eps]
    return {"features": out, "near_constant_dims_scale_1": ignored}


def _flips(q: np.ndarray) -> int:
    return int((np.sum(q[1:] * q[:-1], axis=1) < 0).sum())


def quaternion_audit(episodes: dict[str, dict[str, np.ndarray]], spec: ObservationSpec,
                     low: h5py.File) -> dict[str, Any]:
    """Within-episode sign flips and hemisphere of every quaternion input; dropped rel quats."""
    out: dict[str, Any] = {"spec_features": {}, "dropped_relative_quaternions": {}}
    sl = spec.slices()
    for f in spec.features:
        if f.kind != "body_quat_xyzw":
            continue
        a, b = sl[f.name]
        qs = [episodes[d]["obs"][:, a:b].astype(np.float64) for d in episodes]
        allq = np.concatenate(qs)
        out["spec_features"][f.name] = {
            "within_episode_sign_flips": sum(_flips(q) for q in qs),
            "fraction_w_negative": float((allq[:, 3] < 0).mean()),
            "min_abs_w": float(np.abs(allq[:, 3]).min())}
    T = robosuite_transforms()
    for f in spec.features:
        if f.kind != "pos_in_eef_frame":
            continue
        stored_flips = fresh_flips = n = 0
        for d in demo_keys(low):
            obs = low["data"][d]["obs"]
            p = stored_pose_blocks(spec, f, obs)
            q_stored = obs[f.dataset_ref.key][()][:, f.dataset_ref.hi:f.dataset_ref.hi + 4]
            q_fresh = np.stack([T.mat2pose(T.pose_in_A_to_pose_in_B(
                T.pose2mat((p["obj_pos"][t], p["obj_quat"][t])),
                T.pose_inv(T.pose2mat((p["eef_pos"][t], p["eef_quat"][t])))))[1]
                for t in range(len(q_stored))])
            stored_flips += _flips(q_stored)
            fresh_flips += _flips(q_fresh)
            n += len(q_stored)
        out["dropped_relative_quaternions"][f.name.replace("_pos", "_quat")] = {
            "n_steps": n, "stored_within_episode_sign_flips": stored_flips,
            "same_step_within_episode_sign_flips": fresh_flips,
            "reason_dropped": "robosuite mat2quat forces w >= 0, so the quaternion jumps sign; "
                              "it is a function of robot0_eef_quat and the object quaternion, "
                              "both of which are kept (D-010)"}
    return out


def operator_audit(raw: h5py.File) -> dict[str, Any]:
    """What the raw teleoperation logs say about demonstrators (no identities are present)."""
    channels: dict[str, int] = {}
    interventions = policy_acting = steps = 0
    for d in demo_keys(raw):
        g = raw["data"][d]
        key = ",".join(sorted(g["user_info"].keys()))
        channels[key] = channels.get(key, 0) + 1
        interventions += int(np.sum(g["interventions"][()]))
        policy_acting += int(np.sum(g["policy_acting"][()]))
        steps += len(g["actions"])
    return {"user_info_channel_sets": channels, "intervention_steps": interventions,
            "policy_acting_steps": policy_acting, "steps": steps,
            "operator_identity_available": False,
            "raw_only_groups_unused": sorted(set(raw["data"][demo_keys(raw)[0]].keys())
                                             - {"actions", "states"})}


def window_audit(lengths: dict[str, int], split: dict[str, Any], t_o: int,
                 t_p: int) -> dict[str, Any]:
    """Number of training windows and how many carry padded observations/actions."""
    out: dict[str, Any] = {"obs_horizon": t_o, "pred_horizon": t_p}
    for s, ids in split["splits"].items():
        n = sum(lengths[d] for d in ids)
        pad_obs = sum(min(t_o - 1, lengths[d]) for d in ids)
        pad_act = sum(min(t_p - 1, lengths[d]) for d in ids)
        out[s] = {"n_windows": n, "fraction_with_padded_obs": pad_obs / n,
                  "fraction_with_padded_actions": pad_act / n}
    return out


def closed_loop_audit(env: Any, low: h5py.File, spec: ObservationSpec,
                      demos: list[str]) -> dict[str, Any]:
    """Compare robosuite's ``env.step`` observations with the canonical features in closed loop.

    Replays each demo's actions, then, at every step, records ``max |robosuite obs block -
    feature(state)|`` twice: for the raw step observation, and for robosuite after
    ``sim.forward()`` plus two forced observation updates on a second env restored to the
    same state (current kinematics, fresh cache).
    """
    from roboforge.sim.factory import load_sim_config, make_env
    from roboforge.sim.state_replay import reset_to_state

    extractor = FeatureExtractor(spec)
    ref_env = make_env(load_sim_config(spec.task))
    raw_err: dict[str, float] = {}
    fresh_err: dict[str, float] = {}
    try:
        for d in demos:
            g = low["data"][d]
            reset_to_state(env, g.attrs["model_file"], g["states"][0])
            reset_to_state(ref_env, g.attrs["model_file"], g["states"][0])
            for a in g["actions"][()]:
                ob = env.step(a)[0]
                feats = extractor.dict_from_sim(env.sim)
                ref_env.sim.set_state_from_flattened(env.sim.get_state().flatten())
                ref_env.sim.forward()
                ref_env._get_observations(force_update=True)
                fresh = ref_env._get_observations(force_update=True)
                for f in spec.features:
                    key = "object-state" if f.dataset_ref.key == "object" else f.dataset_ref.key
                    sl = slice(f.dataset_ref.lo, f.dataset_ref.hi)
                    raw_err[f.name] = max(raw_err.get(f.name, 0.0),
                                          float(np.abs(ob[key][sl] - feats[f.name]).max()))
                    fresh_err[f.name] = max(fresh_err.get(f.name, 0.0),
                                            float(np.abs(fresh[key][sl] - feats[f.name]).max()))
    finally:
        ref_env.close()
    return {"demos": demos,
            "max_abs_env_step_obs_minus_feature": raw_err,
            "max_abs_current_robosuite_obs_minus_feature": fresh_err,
            "interpretation": "env.step observations are read from kinematics one physics "
                              "substep old (and, for Can's relative block, one update old). "
                              "With current kinematics and a fresh cache, robosuite equals the "
                              "canonical features exactly, except the corrected relative "
                              "position, which robosuite evaluates with a float32 rotation "
                              "matrix (agreement to float32 precision)"}
