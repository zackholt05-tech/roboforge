"""Render ``docs/DATA_DICTIONARY.md`` from the committed task manifests and the audit report.

The dictionary is generated, never hand-edited, so its tables cannot drift from the data:
every number below comes from ``data/manifests/tasks/*.json`` or ``reports/data_audit.json``.
"""

from __future__ import annotations

from typing import Any


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _obs_table(m: dict[str, Any], a: dict[str, Any]) -> list[str]:
    spec = m["observation_spec"]
    rows = ["| Dims | Feature | Kind | MuJoCo source | Frame | Units | Privileged | "
            "robomimic block | Relation | Train min .. max |", "|" + "---|" * 10]
    feats = a["observations"]["features"]
    for f in spec["features"]:
        lo, hi = spec["slices"][f["name"]]
        src = ", ".join(x for x in (f["site"], f["body"], f["ref_site"], f["ref_body"],
                                    *f["joints"]) if x)
        tr = feats[f["name"]]["train"]
        rng = "; ".join(f"{_fmt(a_)} .. {_fmt(b_)}" for a_, b_ in zip(tr["min"], tr["max"],
                                                                       strict=True))
        ref = f["dataset_ref"]
        rows.append(f"| {lo}-{hi - 1} | `{f['name']}` | {f['kind']} | {src} | {f['frame']} | "
                    f"{f['units']} | {'yes' if f['privileged'] else 'no'} | "
                    f"`{ref['key']}[{ref['lo']}:{ref['hi']}]` | {f['relation']} | {rng} |")
    return rows


def _action_table(m: dict[str, Any], a: dict[str, Any]) -> list[str]:
    s = m["action_spec"]
    tr = a["actions"]["train"]
    rows = ["| Dim | Name | Bounds | Physical scale at ±1 | Train min .. max | Train mean | "
            "Fraction at ±1 |", "|" + "---|" * 7]
    scales = [*(f"±{h} m" for h in s["output_high"][:3]),
              *(f"±{h} rad" for h in s["output_high"][3:6]), "binary: −1 open, +1 close"]
    for i, n in enumerate(s["names"]):
        rows.append(f"| {i} | `{n}` | [{s['low'][i]}, {s['high'][i]}] | {scales[i]} | "
                    f"{_fmt(tr['min'][i])} .. {_fmt(tr['max'][i])} | {_fmt(tr['mean'][i])} | "
                    f"{_fmt(tr['fraction_abs_ge_0.999'][i])} |")
    return rows


def _task_section(m: dict[str, Any], a: dict[str, Any]) -> list[str]:
    ts, ep, q = m["task_spec"], a["episodes"], a["quaternions"]
    files = m["source"]["files"]
    lines = [f"## {m['task']} — `{m['dataset_id']}` ({ts['env_name']}, robosuite "
             f"{ts['env_version']})", "",
             "| Item | Value |", "|---|---|",
             *(f"| Source `{k}` | `{v['path_in_repo']}`, {v['size_bytes']:,} B, SHA-256 "
               f"`{v['sha256']}` |" for k, v in files.items()),
             f"| Episodes | {ep['all']['n_episodes']} ({ep['train']['n_episodes']} train / "
             f"{ep['valid']['n_episodes']} valid), {ep['all']['n_steps']:,} steps |",
             f"| Episode length | {ep['all']['length_steps']['min']:.0f}–"
             f"{ep['all']['length_steps']['max']:.0f} steps (mean "
             f"{ep['all']['length_steps']['mean']:.1f}); "
             f"{ep['all']['duration_s']['min']:.2f}–{ep['all']['duration_s']['max']:.2f} s |",
             f"| Shortest / longest demo | `{ep['shortest']['id']}` ({ep['shortest']['length']}) "
             f"/ `{ep['longest']['id']}` ({ep['longest']['length']}) |",
             f"| Control / physics step | {1 / ts['control_freq_hz']} s / "
             f"{ts['physics_dt_s']} s |",
             f"| State vector | {ts['state_dim']}-D `[time, qpos, qvel]` |",
             f"| Success signal | {ts['success_predicate']} |",
             f"| Evaluation horizon | {ts['eval_horizon_steps']} control steps |",
             f"| Flagged episodes | {len(a['split']['flagged']) or 'none'} |",
             f"| Split | `{m['split']['path']}` (SHA-256 `{m['split']['content_sha256']}`) |",
             f"| Normalization | `{m['normalization']['path']}` (SHA-256 "
             f"`{m['normalization']['content_sha256']}`) |",
             f"| Processed content | `$ROBOFORGE_DATA_ROOT/{m['processed']['relpath']}` (SHA-256 "
             f"`{m['processed']['content_sha256']}`) |",
             f"| Observation spec | {m['observation_spec']['dim']}-D, SHA-256 "
             f"`{m['observation_spec_sha256']}` |",
             f"| Action spec | {m['action_spec']['dim']}-D, SHA-256 `{m['action_spec_sha256']}` |",
             "", "### Observation vector (`obs`, float32)", "", *_obs_table(m, a), "",
             "### Action vector (`actions`, float32)", "", *_action_table(m, a), "",
             "### Quaternion inputs", ""]
    for name, v in q["spec_features"].items():
        lines.append(f"- `{name}`: {v['within_episode_sign_flips']} within-episode sign flips; "
                     f"w < 0 in {v['fraction_w_negative']:.1%} of steps (min |w| "
                     f"{v['min_abs_w']:.2g}).")
    for name, v in q["dropped_relative_quaternions"].items():
        lines.append(f"- `{name}` is **not** an input (dropped, D-010): stored block "
                     f"{v['stored_within_episode_sign_flips']} sign flips, same-step "
                     f"recomputation {v['same_step_within_episode_sign_flips']} flips over "
                     f"{v['n_steps']:,} steps.")
    for name, v in a["ki003_correction"].items():
        lines.append(f"- `{name}` (KI-003 corrected): stored robomimic block minus corrected "
                     f"feature, max {v['stored_minus_corrected_max_t0']:.3g} m at t = 0 and "
                     f"{v['stored_minus_corrected_max_t_gt0']:.3g} m at t > 0; stored block "
                     f"equals robosuite's one-step-lag formula to "
                     f"{v['stored_vs_robosuite_lag_formula_max_t_gt0']:.1e} m. The feature is "
                     f"evaluated in float64; it matches robosuite's float32 formula to "
                     f"{v['max_abs_vs_robosuite_float32_formula']:.1e} m.")
    alts = {"lengths": f"Fraction of {m['task']} train and valid demos per 5-step episode-length "
                       "bin",
            "actions": f"Density of each of the 7 {m['task']} action dimensions, train vs valid",
            "example_episode": f"Feature blocks and actions over time for {m['task']} "
                               f"{a['figures']['example_episode']}"}
    lines += ["", "### Figures (generated; source tables in `reports/data_audit.json`)", ""]
    for path in a["figures"]["files"]:
        kind = path.rsplit("/", 1)[-1].removeprefix(f"{m['task']}_").removesuffix(".png")
        lines.append(f"![{alts[kind]}](../{path})")
    return [*lines, ""]


def render(manifests: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    """Return the Markdown text of the data dictionary."""
    head = [
        "# Data Dictionary", "",
        "> **Generated** by `scripts/audit_data.py` from `data/manifests/tasks/*.json` and "
        "`reports/data_audit.json`. Do not edit by hand; run `make audit-data`.", "",
        "## Conventions (all tasks)", "",
        "- **Observation**: the canonical Level 1 low-dim state vector, computed by "
        "`roboforge.data.features` from the flattened MuJoCo state `[time, qpos, qvel]` with "
        "`mj_kinematics` on a scratch `MjData`. The same function produces dataset windows "
        "and closed-loop evaluation inputs (D-010). Object features are exact simulator state: "
        "**privileged** relative to camera-based policies.",
        "- **Quaternions** are `xyzw` (robosuite convention); MuJoCo `qpos`/`xquat` are `wxyz`. "
        "`robot0_eef_quat` is the orientation of the `robot0_right_hand` *body*, while "
        "`robot0_eef_pos` is the `gripper0_right_grip_site` position (robosuite ≤ 1.5 "
        "behaviour, robosuite issue #298).",
        "- **World frame**: MuJoCo world, metres, z up.",
        "- **Actions**: OSC_POSE deltas in the world frame (`input_ref_frame=world`) plus a "
        "binary gripper command; 20 Hz; stored values already lie in [−1, 1].",
        "- **Windows** (`roboforge.data.sequence.SequenceDataset`): `obs_history[T_o, D]`, "
        "`obs_valid[T_o]`, `future_action[T_p, A]`, `valid_mask[T_p]`, `episode_id`, "
        "`start_t`. One window per decision step; the first observation is repeated before "
        "the episode start (`obs_valid=False`) and the last action after its end "
        "(`valid_mask=False`, excluded from losses). Defaults T_o = 2, T_p = 16.",
        "- **Normalization**: per-dimension min-max to [−1, 1], fit on per-step arrays of "
        "training episodes only; dimensions with training range < 1e-4 use scale 1 (D-020). "
        "Model actions are decoded by inverting the transform and clipping to the action "
        "bounds.",
        "- **Splits**: robomimic upstream `mask/train` / `mask/valid` (D-019). No "
        "demonstration test split exists: final tests use held-out simulator initial states.",
        "- **Operator metadata**: none. The raw files hold one teleoperation channel per demo "
        "and no demonstrator identity, so no held-out-operator claims are possible.", ""]
    body = [line for m in manifests for line in _task_section(m, audit["tasks"][m["task"]])]
    return "\n".join(head + body).rstrip() + "\n"
