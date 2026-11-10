"""Stage the Observatory build for GitHub Pages, or verify the live site (DECISIONS D-102).

    uv run --frozen python scripts/pages_bundle.py stage --out logs/tmp/pages
    uv run --frozen python scripts/pages_bundle.py verify --url https://<owner>.github.io/<repo>/ \
        --manifest logs/tmp/pages/deploy.json

`stage` refuses a dirty tree: the hosted build must come from one committed revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.paths import REPO_ROOT
from roboforge.provenance import git_revision
from roboforge.release import pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--allow-dirty", action="store_true", help="development only")
    v = sub.add_parser("verify")
    v.add_argument("--url", required=True)
    v.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "stage":
        git = git_revision()
        if git.get("dirty") and not a.allow_dirty:
            print("refusing: the working tree is dirty (D-102)", file=sys.stderr)
            return 2
        out = a.out if a.out.is_absolute() else REPO_ROOT / a.out
        m = pages.stage(out, git)
        total = sum(f["bytes"] for f in m["files"].values())
        print(f"staged {len(m['files'])} files, {total / 1e6:.1f} MB, from {git.get('sha')}")
        return 0
    problems = pages.verify(a.url, json.loads(a.manifest.read_text()))
    n = len(json.loads(a.manifest.read_text())["files"])
    print("\n".join(problems) or f"live site matches deploy.json ({n} files, .nojekyll not served)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
