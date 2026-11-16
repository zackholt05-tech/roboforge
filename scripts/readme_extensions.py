"""Write (or check) the README's generated extensions block (DECISIONS D-133).

    uv run --frozen python scripts/readme_extensions.py          # rewrite the block
    uv run --frozen python scripts/readme_extensions.py --check  # fail if it is stale

The numbers come only from paper/addendum/generated/numbers.tex (make addendum-check).
"""

from __future__ import annotations

import argparse
import sys

from roboforge.addendum import readme
from roboforge.paths import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="only compare, never write")
    a = p.parse_args(argv)
    if a.check:
        problems = readme.check()
        print("\n".join(problems) or "README extensions block and chart match a fresh build")
        return 1 if problems else 0
    path = REPO_ROOT / readme.README
    path.write_text(readme.render(path.read_text(encoding="utf-8"), readme.load()),
                    encoding="utf-8")
    print(f"wrote the generated {readme.NAME} block of {readme.README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
