"""X2 (Can MH): read-only inspection of the downloaded MH files (DECISIONS D-118).

Nothing about the MH data is assumed (``docs/extensions/X2_harder_task.md`` §3). From the
hash-verified files only, :func:`inspect_mh` reads:

* the episode count and lengths (control steps), overall and per operator;
* every ``mask/`` filter key and its size; which keys name operators; whether the operator
  masks partition the demos; whether the upstream ``train``/``valid`` masks are the union of the
  per-operator ``<operator>_train``/``<operator>_valid`` masks (operator-stratified);
* the raw file's per-demo teleoperation arrays (every dataset the low-dim file lacks), with
  their counts, as the remaining operator evidence;
* ``env_args`` of both files against the validated Can simulator config
  (``configs/simulator/can.yaml``, task-defining kwargs) and the Can PH low-dim file;
* robomimic's registered evaluation horizon for Can MH (robomimic@``d309eae`` registry);
* whether the low-dim and raw files hold identical states, actions, model XML and masks.

It writes nothing and decides nothing: the split (D-119) and the episode horizon are decided
from this report, before any training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from roboforge.data.inspect import demo_keys, env_args

OPERATOR_MASK = re.compile(r"^(better|okay|worse)_operator_(\d+)$")


def _idx(d: str) -> int:
    return int(d.split("_")[-1])


def masks(f: h5py.File) -> dict[str, list[str]]:
    """Every ``mask/<key>`` as a list of demo IDs sorted by index."""
    if "mask" not in f:
        return {}
    return {k: sorted((x.decode() if isinstance(x, bytes) else str(x) for x in f["mask"][k][()]),
                      key=_idx) for k in sorted(f["mask"].keys())}


def operator_masks(m: dict[str, list[str]]) -> dict[str, list[str]]:
    """The masks that name one operator (``<quality>_operator_<n>``)."""
    return {k: v for k, v in m.items() if OPERATOR_MASK.match(k)}


def length_stats(lengths: list[int]) -> dict[str, Any]:
    """Summary of episode lengths in control steps."""
    a = np.asarray(lengths)
    names = ("p10", "p25", "p50", "p75", "p90", "p95", "p99")
    q = [float(x) for x in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])]
    return {"n": int(a.size), "total": int(a.sum()), "min": int(a.min()), "max": int(a.max()),
            "mean": float(a.mean()), "quantiles": dict(zip(names, q, strict=True)),
            "n_over_400": int((a > 400).sum()), "n_over_500": int((a > 500).sum())}


def operator_report(demos: list[str], m: dict[str, list[str]],
                    lengths: dict[str, int]) -> dict[str, Any]:
    """Operator masks: partition check, per-operator lengths and train/valid stratification."""
    ops = operator_masks(m)
    owner: dict[str, list[str]] = {}
    for op, ids in ops.items():
        for d in ids:
            owner.setdefault(d, []).append(op)
    multi = sorted((d for d, o in owner.items() if len(o) > 1), key=_idx)
    unassigned = sorted(set(demos) - set(owner), key=_idx)
    per_op = {}
    for op, ids in ops.items():
        tr, va = m.get(f"{op}_train"), m.get(f"{op}_valid")
        per_op[op] = {
            "n": len(ids), "demo_index_range": [_idx(ids[0]), _idx(ids[-1])],
            "lengths": length_stats([lengths[d] for d in ids]),
            "train_mask": len(tr) if tr is not None else None,
            "valid_mask": len(va) if va is not None else None,
            "in_upstream_train": len(set(ids) & set(m.get("train", []))),
            "in_upstream_valid": len(set(ids) & set(m.get("valid", []))),
        }
    union_tr = set().union(*(m.get(f"{op}_train", []) for op in ops)) if ops else set()
    union_va = set().union(*(m.get(f"{op}_valid", []) for op in ops)) if ops else set()
    tiers = {}
    for tier in ("better", "okay", "worse"):
        members = [op for op in ops if op.startswith(f"{tier}_")]
        tiers[tier] = {"operators": members,
                       "mask_equals_union_of_operators": (
                           sorted(set(m.get(tier, []))) ==
                           sorted(set().union(*(ops[o] for o in members))) if members else None)}
    return {
        "operator_masks": sorted(ops),
        "n_operators": len(ops),
        "partition_all_demos": not multi and not unassigned,
        "demos_in_several_operator_masks": multi,
        "demos_in_no_operator_mask": unassigned,
        "per_operator": per_op,
        "quality_tiers": tiers,
        "upstream_train_equals_union_of_operator_train": (
            sorted(union_tr) == sorted(set(m.get("train", [])))),
        "upstream_valid_equals_union_of_operator_valid": (
            sorted(union_va) == sorted(set(m.get("valid", [])))),
        "upstream_train_valid_partition": (
            not set(m.get("train", [])) & set(m.get("valid", []))
            and set(m.get("train", [])) | set(m.get("valid", [])) == set(demos)),
    }


def _ds_names(g: h5py.Group, prefix: str = "") -> list[str]:
    out = []
    for k, v in g.items():
        p = f"{prefix}{k}"
        out.extend(_ds_names(v, p + "/") if isinstance(v, h5py.Group) else [p])
    return out


def teleop_report(raw: h5py.File, low: h5py.File) -> dict[str, Any]:
    """Per-demo datasets and attributes of the raw file that the low-dim file lacks.

    For boolean arrays: the number of demos with any true entry and the total true count; for
    others: dtype and shapes. These are the remaining places operator identity could live."""
    demos = demo_keys(raw)
    names: set[str] = set()
    attrs: set[str] = set()
    for d in demos:
        names |= set(_ds_names(raw["data"][d]))
        attrs |= set(raw["data"][d].attrs.keys())
    low_names = set(_ds_names(low["data"][demo_keys(low)[0]]))
    low_attrs = set(low["data"][demo_keys(low)[0]].attrs.keys())
    out: dict[str, Any] = {}
    for n in sorted(names - low_names):
        arrs = [raw["data"][d][n][()] for d in demos if n in raw["data"][d]]
        a0 = np.asarray(arrs[0])
        rec: dict[str, Any] = {"dtype": str(a0.dtype), "n_demos_with_key": len(arrs),
                               "ndim": int(a0.ndim)}
        if a0.dtype == bool:
            rec["n_demos_any_true"] = int(sum(bool(np.asarray(a).any()) for a in arrs))
            rec["n_demos_all_true"] = int(sum(bool(np.asarray(a).all()) for a in arrs))
            rec["true_steps_total"] = int(sum(int(np.asarray(a).sum()) for a in arrs))
        out[n] = rec
    return {"raw_only_datasets": out, "raw_only_demo_attrs": sorted(attrs - low_attrs),
            "low_only_demo_attrs": sorted(low_attrs - attrs),
            "raw_file_attrs": sorted(raw.attrs.keys()),
            "raw_data_attrs": sorted(raw["data"].attrs.keys())}


def low_raw_equality(low: h5py.File, raw: h5py.File) -> dict[str, Any]:
    """Do the two files hold the same demos, states, actions, model XML and masks?"""
    dl, dr = demo_keys(low), demo_keys(raw)
    same = dl == dr
    s_err = a_err = 0.0
    xml_diff = []
    if same:
        for d in dl:
            gl, gr = low["data"][d], raw["data"][d]
            s_err = max(s_err, float(np.abs(gl["states"][()] - gr["states"][()]).max()))
            a_err = max(a_err, float(np.abs(gl["actions"][()] - gr["actions"][()]).max()))
            if gl.attrs["model_file"] != gr.attrs["model_file"]:
                xml_diff.append(d)
    return {"same_demo_keys": same, "states_max_abs_diff": s_err, "actions_max_abs_diff": a_err,
            "model_xml_differs": xml_diff, "masks_equal": masks(low) == masks(raw)}


def _env_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}{k}: only in {'second' if k not in a else 'first'}")
            else:
                out.extend(_env_diff(a[k], b[k], f"{path}{k}."))
        return out
    return [] if a == b else [f"{path.rstrip('.')}: {a!r} != {b!r}"]


def env_report(low: h5py.File, raw: h5py.File, ph_low: h5py.File | None) -> dict[str, Any]:
    """``env_args`` of the MH files against the Can simulator config and the Can PH file."""
    from roboforge.sim.factory import load_sim_config, task_kwargs

    sim = load_sim_config("can")
    out: dict[str, Any] = {}
    for name, f in (("low_dim", low), ("raw", raw)):
        ea = env_args(f)
        out[name] = {
            "env_name": ea["env_name"], "env_version": ea["env_version"], "type": ea["type"],
            "task_kwargs_vs_can_sim_config": _env_diff(task_kwargs(ea), sim["env_kwargs"]),
            "env_name_equal": ea["env_name"] == sim["env_name"],
            "type_equal": ea["type"] == sim["type"],
        }
        if ph_low is not None:
            out[name]["env_args_vs_can_ph_low_dim"] = _env_diff(ea, env_args(ph_low))
    out["can_sim_config_env_version"] = sim["env_version"]
    return out


def registry_horizon() -> dict[str, Any]:
    """robomimic's registered evaluation horizons for Can MH and Can PH (installed package)."""
    import robomimic

    reg = robomimic.DATASET_REGISTRY
    return {"can_mh_low_dim": reg["can"]["mh"]["low_dim"]["horizon"],
            "can_mh_link": reg["can"]["mh"]["low_dim"]["url"],
            "can_ph_low_dim": reg["can"]["ph"]["low_dim"]["horizon"],
            "robomimic_version": getattr(robomimic, "__version__", None)}


def inspect_mh(low_path: Path, raw_path: Path, ph_low_path: Path | None) -> dict[str, Any]:
    """The full inspection report (JSON-compatible)."""
    with h5py.File(low_path, "r") as low, h5py.File(raw_path, "r") as raw:
        ph = h5py.File(ph_low_path, "r") if ph_low_path and ph_low_path.exists() else None
        try:
            demos = demo_keys(low)
            lengths = {d: int(low["data"][d].attrs["num_samples"]) for d in demos}
            m = masks(low)
            first = low["data"][demos[0]]
            report = {
                "n_demos": len(demos), "data_total": int(low["data"].attrs["total"]),
                "lengths": length_stats(list(lengths.values())),
                "state_dim": int(first["states"].shape[1]),
                "action_dim": int(first["actions"].shape[1]),
                "obs_keys": {k: list(first["obs"][k].shape[1:]) for k in first["obs"]},
                "demo_attrs_low_dim": sorted(first.attrs.keys()),
                "masks": {k: len(v) for k, v in m.items()},
                "operators": operator_report(demos, m, lengths),
                "teleoperation": teleop_report(raw, low),
                "low_dim_vs_raw": low_raw_equality(low, raw),
                "env_args": env_report(low, raw, ph),
                "registry_horizon": registry_horizon(),
                "env_args_low_dim": env_args(low),
            }
            if ph is not None:
                ph_len = [int(ph["data"][d].attrs["num_samples"]) for d in demo_keys(ph)]
                report["can_ph_lengths"] = length_stats(ph_len)
        finally:
            if ph is not None:
                ph.close()
    return json.loads(json.dumps(report, default=str))
