"""Generate or check the paper's tables, figures and numbers (DECISIONS D-094).

    uv run --frozen python scripts/paper_assets.py build   # write paper/generated/
    uv run --frozen python scripts/paper_assets.py check   # rebuild in memory, compare
    uv run --frozen python scripts/paper_assets.py lint    # LaTeX: keys defined, no bare numbers

Every output is computed from hash-pinned frozen files (``roboforge.paper.sources``); nothing
is evaluated, trained or recomputed statistically. ``check`` exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import sys

from roboforge.paper import build, lint


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["build", "check", "lint"])
    args = ap.parse_args(argv)
    if args.command == "lint":
        problems = lint.lint_paper()
        print("\n".join(problems) or "paper lint: every number comes from a generated macro")
        return 1 if problems else 0
    files = build.build_all()
    if args.command == "build":
        written = build.write(files)
        print(f"wrote {len(written)} files under {build.OUT}/")
        return 0
    problems = build.check(files)
    print("\n".join(problems) or f"{build.OUT}/ matches a fresh build ({len(files)} files)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
