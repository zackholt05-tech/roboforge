"""Build or check the v1.1 addendum's generated files and the README chart (DECISIONS D-133).

    uv run --frozen python scripts/addendum_assets.py build   # write paper/addendum/generated/
    uv run --frozen python scripts/addendum_assets.py check   # fail unless equal to a fresh build
    uv run --frozen python scripts/addendum_assets.py lint    # macros defined, no bare numbers

Every value comes from the three hash-pinned extension results (roboforge.addendum.sources).
"""

from __future__ import annotations

import argparse
import sys

from roboforge.addendum import build


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["build", "check", "lint"])
    a = p.parse_args(argv)
    if a.command == "lint":
        problems = build.lint()
        print("\n".join(problems) or "addendum lint: every number comes from a generated macro")
        return 1 if problems else 0
    files = build.build_all()
    if a.command == "build":
        written = build.write(files)
        print(f"wrote {len(written)} files")
        return 0
    problems = build.check(files)
    print("\n".join(problems) or f"{len(files)} addendum files equal a fresh build")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
