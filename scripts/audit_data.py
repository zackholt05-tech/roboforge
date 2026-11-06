"""Audit the canonical Lift/Can data pipeline and write its summaries.

For each task the script first rebuilds the pipeline in memory and requires it to equal the
committed artifacts (it fails otherwise). It then computes the dataset audit from the
hash-verified source files and the rebuilt arrays, and writes:

* ``reports/data_audit.json``: every audit number, with an environment snapshot;
* ``reports/data_audit/<task>_{lengths,actions,example_episode}.png``: explorer figures;
* ``docs/DATA_DICTIONARY.md``: generated from the task manifests and the audit.

Usage::

    python scripts/audit_data.py --tasks lift can
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import h5py  # noqa: E402

from roboforge.data import audit as A  # noqa: E402
from roboforge.data import explore  # noqa: E402
from roboforge.data.dictionary import render  # noqa: E402
from roboforge.data.fetch import load_manifest  # noqa: E402
from roboforge.data.inspect import demo_keys  # noqa: E402
from roboforge.data.normalization import Normalizer  # noqa: E402
from roboforge.data.pipeline import build_task, check_task, load_task_artifacts  # noqa: E402
from roboforge.data.specs import build_observation_spec, load_data_config  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, write_json  # noqa: E402
from roboforge.sim.factory import load_sim_config, make_env  # noqa: E402

CLOSED_LOOP_DEMOS = 5  # first N training demos (numeric order) replayed for the obs-drift audit


def audit_task(task: str, fig_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(task_manifest, audit)`` for one task; raise if the rebuild != committed."""
    art = build_task(task)
    problems = check_task(art)
    if problems:
        raise SystemExit(f"[{task}] committed artifacts out of date: {problems}")
    docs = load_task_artifacts(task)
    m, split, norm = docs["task_manifest"], docs["split"], docs["normalization"]
    _, files = load_manifest()
    path = {f.hdf5_type: f.destination() for f in files if f.task == task}
    lengths = {e["id"]: e["length"] for e in m["episodes"]}
    dt = 1.0 / m["task_spec"]["control_freq_hz"]
    obs_norm = Normalizer.from_dict(norm["obs"])
    names = list(m["action_spec"]["names"])
    eps = art.episodes
    figs = [explore.length_histogram(lengths, split, task, fig_dir / f"{task}_lengths.png"),
            explore.action_distributions(eps, split, names, task, fig_dir / f"{task}_actions.png")]
    slices = {k: tuple(v) for k, v in m["observation_spec"]["slices"].items()}
    ordered = dict(sorted(slices.items(), key=lambda kv: kv[1][0]))
    fig, example = explore.example_episode(eps, split, ordered, names, task,
                                           fig_dir / f"{task}_example_episode.png")
    figs.append(fig)
    with h5py.File(path["low_dim"], "r") as low, h5py.File(path["raw"], "r") as raw:
        spec = build_observation_spec(load_data_config(task), low)
        if spec.sha256() != m["observation_spec_sha256"]:
            raise SystemExit(f"[{task}] observation spec drifted from the committed manifest")
        env = make_env(load_sim_config(task))
        try:
            demos = split["splits"]["train"][:CLOSED_LOOP_DEMOS]
            closed = A.closed_loop_audit(env, low, spec, demos)
        finally:
            env.close()
        final_ok = art.verification.final_state_success
        xml_unique = len({low["data"][d].attrs["model_file"] for d in demo_keys(low)})
        report = {
            "episodes": A.episode_audit(lengths, split, dt),
            "time_and_success": A.time_and_success_audit(low),
            "final_state_success": {"n_demos": len(final_ok), "n_success": sum(final_ok.values())},
            "sanity": {"checks": ["array lengths == num_samples", "finite values",
                                  "actions within bounds", f"state time step == {dt} s",
                                  "obs[t+1] == next_obs[t]", "final reward/done == 1",
                                  "saved final state satisfies the success predicate"],
                       "flagged": art.issues},
            "split": {k: split[k] for k in ("content_sha256", "counts", "samples", "flagged",
                                            "method")} | {"valid_ids": split["splits"]["valid"]},
            "actions": A.action_audit(eps, split, names),
            "observations": A.feature_audit(eps, split, spec, obs_norm),
            "quaternions": A.quaternion_audit(eps, spec, low),
            "feature_vs_dataset_max_abs": art.verification.max_abs_vs_dataset,
            "ki003_correction": art.verification.corrected,
            "closed_loop_robosuite_obs_vs_features": closed,
            "model_xml_unique_count": xml_unique,
            "operator": A.operator_audit(raw),
            "windows": A.window_audit(lengths, split, m["windows"]["obs_horizon"],
                                      m["windows"]["pred_horizon"]),
            "normalization": {"content_sha256": norm["content_sha256"],
                              "obs_mode": norm["obs"]["mode"],
                              "action_mode": norm["action"]["mode"],
                              "action_scale": norm["action"]["scale"],
                              "action_offset": norm["action"]["offset"]},
            "hashes": {"processed_content_sha256": m["processed"]["content_sha256"],
                       "observation_spec_sha256": m["observation_spec_sha256"],
                       "action_spec_sha256": m["action_spec_sha256"],
                       "task_spec_sha256": m["task_spec_sha256"],
                       "task_manifest_sha256": m["content_sha256"]},
            "figures": {"files": [repo_relative(p) for p in figs], "example_episode": example},
        }
    return m, report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "data_audit.json")
    p.add_argument("--dictionary", type=Path, default=REPO_ROOT / "docs" / "DATA_DICTIONARY.md")
    args = p.parse_args(argv)
    fig_dir = args.out.parent / "data_audit"
    manifests, tasks = [], {}
    for task in args.tasks:
        m, report = audit_task(task, fig_dir)
        manifests.append(m)
        tasks[task] = report
        cl = report["closed_loop_robosuite_obs_vs_features"]
        raw = max(cl["max_abs_env_step_obs_minus_feature"].values())
        cur = max(cl["max_abs_current_robosuite_obs_minus_feature"].values())
        print(f"[{task}] audit done; flagged {report['sanity']['flagged'] or 'none'}; "
              f"env.step obs drift max {raw:.2e}; current-robosuite max {cur:.2e}")
    audit = {"schema": "roboforge.data_audit/1", "generated_by": "scripts/audit_data.py",
             "environment": environment_snapshot(), "tasks": tasks}
    write_json(args.out, audit)
    args.dictionary.write_text(render(manifests, audit))
    print(f"wrote {repo_relative(args.out)}, {repo_relative(args.dictionary)}, figures in "
          f"{repo_relative(fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
