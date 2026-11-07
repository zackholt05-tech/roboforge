"""Label recorded episodes with the preregistered failure taxonomy (RESEARCH_PROTOCOL §14.5).

    python scripts/failure_taxonomy.py --summaries reports/session7/robustness/evals/*.json \
        --out reports/session7/taxonomy/robustness.json

Reads each evaluation summary, labels every episode record under its ``records_dir`` with
:func:`roboforge.eval.taxonomy.classify` and writes one report: the taxonomy config and its
hash, then per evaluation the category counts (every category, zeros included) and the
per-episode labels with their attributes. It selects nothing and changes no record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.eval import taxonomy as tx
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import git_revision, sha256_file, write_json


def label_summary(path: Path) -> dict:
    """Taxonomy counts and labels of one evaluation summary's episode records."""
    s = json.loads(path.read_text())
    labels = tx.classify_records(REPO_ROOT / s["records_dir"])
    if len(labels) != s["n_episodes"]:
        raise ValueError(f"{path}: {len(labels)} records for {s['n_episodes']} episodes")
    succ = sum(lab["category"] == "success" for lab in labels)
    if succ != s["successes"]:
        raise ValueError(f"{path}: taxonomy successes {succ} != summary {s['successes']}")
    pol = s["policy"]["params"]
    return {"summary": repo_relative(path), "summary_sha256": sha256_file(path),
            "eval_run_id": s["eval_run_id"], "train_run_id": s.get("train_run_id"),
            "family": pol.get("family", s["policy"]["name"]), "step": s.get("checkpoint", {})
            .get("step"), "task": s["task"], "suite": s["suite"],
            "inference_variant": s.get("inference_variant"),
            "counts": tx.counts(labels), "labels": labels}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summaries", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    evals = [label_summary(s) for s in sorted(a.summaries)]
    write_json(a.out, {"schema": "roboforge.failure_taxonomy_report/1",
                       "taxonomy": tx.taxonomy_config(),
                       "taxonomy_sha256": tx.taxonomy_sha256(),
                       "n_evaluations": len(evals),
                       "n_episodes": sum(len(e["labels"]) for e in evals),
                       "evaluations": evals, "git": git_revision()})
    print(f"labelled {sum(len(e['labels']) for e in evals)} episodes of {len(evals)} "
          f"evaluations -> {repo_relative(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
