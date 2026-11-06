"""Export real denoising traces of a diffusion policy's recorded validation decisions.

    ROBOFORGE_EVIDENCE_SESSION=4 infra/containers/run_sim_cpu.sh \
        python scripts/export_denoising.py --eval-run runs/<eval_run_id>

Reads one closed-loop validation evaluation of a diffusion checkpoint (its immutable
manifest and episode records), reloads the checkpoint by SHA-256 and replays the selected
episodes decision by decision with their recorded noise seeds
(:mod:`roboforge.viz.denoising`). Every replayed decision must reproduce the recorded actions
bitwise, so run it on the platform that recorded the evaluation.

**Selection rule (declared before export, not by appearance).** Episodes: the first
successful and the first failed episode in sorted ``state_id`` order (``--states``
overrides). Decisions per episode: the first, the middle (``n // 2``) and the last. For each
selected decision, ``--diversity`` extra samples (seeds ``1_000_000 + i``) are drawn from the
same conditioning window; they are labeled never-executed proposals.

Writes ``<out-dir>/<eval_run_id>/<state_id>_d<decision>.{json,npz}`` and an ``index.json``
(default out-dir: ``reports/session$ROBOFORGE_EVIDENCE_SESSION/denoising``, D-039).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.data.pipeline import load_specs  # noqa: E402
from roboforge.eval.learned import LearnedPolicy  # noqa: E402
from roboforge.eval.rollout import RolloutRecord  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file  # noqa: E402
from roboforge.viz.denoising import (  # noqa: E402
    SCHEMA,
    decision_frames,
    decision_trace,
    diversity_samples,
    replay_episode,
)

DIVERSITY_SEED_BASE = 1_000_000


def select_states(episodes: list[dict]) -> list[str]:
    """First success and first failure in sorted state_id order."""
    ordered = sorted(episodes, key=lambda e: e["state_id"])
    out = []
    for want in (True, False):
        hit = next((e["state_id"] for e in ordered if e["success"] == want), None)
        if hit is not None:
            out.append(hit)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval-run", type=Path, required=True)
    p.add_argument("--states", nargs="+", default=None)
    p.add_argument("--diversity", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    out_root = (args.out_dir or evidence_reports_dir("denoising"))
    man = json.loads((args.eval_run / "manifest.json").read_text())
    ck = man["checkpoint"]
    params = man["policy"]["params"]
    policy = LearnedPolicy.from_checkpoint(REPO_ROOT / ck["path"], expected_sha256=ck["sha256"],
                                           policy_seed=params["policy_seed"],
                                           repeat=params["repeat"])
    _, action_spec = load_specs(man["task"])
    codec = policy.codec(action_spec)
    states = args.states or select_states(man["episodes"])
    out_dir = out_root / man["eval_run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sid in states:
        rec = RolloutRecord.load(args.eval_run / "episodes" / sid)
        frames = decision_frames(rec)
        n = len(frames)
        capture = sorted({0, n // 2, n - 1})
        caps = replay_episode(policy, rec, set(capture))
        for d in capture:
            meta, arrays = decision_trace(policy, rec, caps[d], codec)
            if args.diversity:
                div = diversity_samples(policy, caps[d], args.diversity,
                                        DIVERSITY_SEED_BASE)
                arrays["diversity_chunks_normalized"] = div["chunks_normalized"]
                arrays["diversity_std_per_step_dim"] = div["std_over_samples_per_step_dim"]
                meta["diversity"] = {"seeds": div["seeds"], "label": div["label"],
                                     "mean_std_normalized": float(
                                         div["std_over_samples_per_step_dim"].mean())}
            meta.update(platform_exported=evaluation_platform(), git=git_revision(),
                        checkpoint=ck, selection_rule=("first success and first failure by "
                                                       "sorted state_id; decisions first, "
                                                       "middle, last"))
            stem = out_dir / f"{sid}_d{d:03d}"
            np.savez_compressed(stem.with_suffix(".npz"), **arrays)
            meta["arrays_file"] = {"path": repo_relative(stem.with_suffix(".npz")),
                                   "sha256": sha256_file(stem.with_suffix(".npz"))}
            stem.with_suffix(".json").write_text(json.dumps(meta, indent=1, allow_nan=False)
                                                 + "\n")
            index.append({"state_id": sid, "decision": d, "success": rec.outcome["success"],
                          "n_decisions": n, "json": repo_relative(stem.with_suffix(".json")),
                          "verified": meta["verification"]})
            print(f"[{sid} decision {d}/{n - 1}] {meta['n_steps']} sampler steps, executed "
                  f"{meta['n_executed']} actions; replay bitwise = recorded; "
                  f"-> {repo_relative(stem.with_suffix('.json'))}")
    (out_dir / "index.json").write_text(json.dumps(
        {"schema": SCHEMA + "/index", "eval_run_id": man["eval_run_id"],
         "eval_manifest_sha256": sha256_file(args.eval_run / "manifest.json"),
         "checkpoint": ck, "platform": evaluation_platform(), "git": git_revision(),
         "entries": index}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
