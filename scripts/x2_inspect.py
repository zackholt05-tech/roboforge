"""X2: inspect the hash-verified Can MH files and write the inspection report (D-118).

Reads ``raw/robomimic_v1.5/can/mh/{low_dim_v15,demo_v15}.hdf5`` (verified against
``data/manifests/robomimic_v1.5_mh.json`` first) and the Can PH low-dim file for comparison,
and writes ``reports/session<N>/x2/inspection.json`` (:func:`roboforge.x2.inspect.inspect_mh`).
Read-only; decides nothing::

    ROBOFORGE_EVIDENCE_SESSION=15 python scripts/x2_inspect.py
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.data.fetch import load_manifest, verify_file
from roboforge.paths import evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, write_json
from roboforge.x2.data import X2_MANIFEST
from roboforge.x2.inspect import inspect_mh


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.parse_args(argv)
    _, files = load_manifest(X2_MANIFEST)
    paths = {}
    for f in files:
        paths[f.hdf5_type] = f.destination()
        verify_file(f, paths[f.hdf5_type])
    _, ph_files = load_manifest()
    ph = next(f for f in ph_files
              if (f.task, f.dataset_type, f.hdf5_type) == ("can", "ph", "low_dim"))
    verify_file(ph, ph.destination())
    report = {"schema": "roboforge.x2_inspection/1", "decision": "D-118",
              "manifest": X2_MANIFEST,
              "files": {k: {"sha256": next(f.sha256 for f in files if f.hdf5_type == k)}
                        for k in paths},
              "can_ph_low_dim_sha256": ph.sha256, "git": git_revision(),
              **inspect_mh(paths["low_dim"], paths["raw"], ph.destination())}
    out = evidence_reports_dir("x2") / "inspection.json"
    write_json(out, report)
    ops = report["operators"]
    print(json.dumps({"n_demos": report["n_demos"], "lengths": report["lengths"],
                      "n_operators": ops["n_operators"],
                      "operators_partition": ops["partition_all_demos"],
                      "upstream_split_stratified": (
                          ops["upstream_train_equals_union_of_operator_train"]
                          and ops["upstream_valid_equals_union_of_operator_valid"]),
                      "registry_horizon": report["registry_horizon"],
                      "low_dim_vs_raw": report["low_dim_vs_raw"],
                      "out": repo_relative(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
