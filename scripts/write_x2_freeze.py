"""Write ``docs/extensions/X2_FREEZE.md`` from committed files, once (DECISIONS D-130).

    ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python scripts/write_x2_freeze.py \\
        --decisions-commit <sha of the D-129/D-130 commit>

Run from a clean tree whose ``HEAD`` is the X2-c code commit; it refuses if the freeze file
already exists. The block pins the X2-b test configuration (checked by its SHA-256), the
held-out suite's content hash (read from its committed open manifest; no state is loaded), the
analysis design (:func:`roboforge.x2.analysis.design`) and the SHA-256 of every pinned file
(:data:`roboforge.x2.freeze_doc.CODE_FILES`). The written document is parsed back before the
script exits. The file is then committed alone, before any X2 held-out rollout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime

import yaml

from roboforge.paths import REPO_ROOT, configs_dir
from roboforge.provenance import git_blob_sha1, sha256_file
from roboforge.sim import eval_states as es
from roboforge.x2 import freeze_doc as fd
from roboforge.x2 import heldout as xh
from roboforge.x2 import states as x2s
from roboforge.x2.host import tree_clean


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def _last_commit(rel: str) -> str:
    return _git("log", "-1", "--format=%h", "--", rel)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--decisions-commit", required=True)
    a = p.parse_args(argv)
    path = REPO_ROOT / xh.FREEZE_REL
    if path.exists():
        raise SystemExit(f"{xh.FREEZE_REL} exists: the X2 freeze is written once")
    if not tree_clean():
        raise SystemExit("the working tree is dirty: write the freeze from the clean code commit")
    dec = _git("rev-parse", "--short", a.decisions_commit)
    if subprocess.run(["git", "merge-base", "--is-ancestor", dec, "HEAD"],
                      cwd=REPO_ROOT).returncode != 0:
        raise SystemExit(f"{dec} is not an ancestor of HEAD")
    if "### D-130 " not in _git("show", f"{dec}:docs/DECISIONS.md"):
        raise SystemExit(f"{dec} does not record D-130")
    tc_text, tc = fd.load_test_configuration()
    manifest = es.read_manifest(x2s.suite_task(), x2s.HELDOUT_SUITE)  # open manifest: hashes
    platform = yaml.safe_load((configs_dir() / "eval" / "platform.yaml").read_text())
    code = _git("rev-parse", "--short", "HEAD")
    commit_order = {
        "preregistration": f"{_last_commit(xh.PREREGISTRATION)} "
                           "(docs/extensions/X2_PREREGISTRATION.md, before any X2-b training)",
        "x2b_evidence": f"{_last_commit(xh.TEST_CONFIGURATION)} (the X2 test configuration)",
        "decisions": f"{dec} (D-129, D-130, before any X2-c code)",
        "code": f"{code} (the X2-c code and tests; the pinned hashes below)",
        "freeze": "the commit that adds this file, alone, from a clean tree, before any X2 "
                  "held-out rollout",
        "evaluations": "the 12 evaluations, once, from the clean freeze commit; their records "
                       "committed after",
        "analysis": "once, from a clean commit holding this file and the 12 summaries",
    }
    block = fd.build_block(tc, manifest, platform, sha256_file(REPO_ROOT / xh.PREREGISTRATION),
                           commit_order, fd.code_hashes())
    frozen_on = datetime.now(tz=UTC).strftime("%Y-%m-%d (UTC)")
    text = fd.render(block, tc_text, frozen_on)
    if xh.parse_x2_freeze(text)["families"] != block["families"]:
        raise SystemExit("the rendered freeze does not parse back to its block")
    with path.open("x") as fh:
        fh.write(text)
    print(f"-> {xh.FREEZE_REL} (sha256 {sha256_file(path)}, git blob "
          f"{git_blob_sha1(path.read_bytes())}); commit it alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
