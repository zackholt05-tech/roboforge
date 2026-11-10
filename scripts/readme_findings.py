"""Write (or check) the README's generated key-findings block (DECISIONS D-103).

    uv run --frozen python scripts/readme_findings.py          # rewrite README.md's block
    uv run --frozen python scripts/readme_findings.py --check  # fail if it is stale

The numbers come only from paper/generated/numbers.tex (verified by make paper-check).
"""

from __future__ import annotations

import argparse
import sys

from roboforge.release.macros import Macros, load_macros
from roboforge.release.readme import README, check_readme, findings_block, replace_block


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="only compare, never write")
    a = p.parse_args(argv)
    if a.check:
        problems = check_readme()
        print("\n".join(problems) or "README findings block matches a fresh generation")
        return 1 if problems else 0
    text = README.read_text(encoding="utf-8")
    README.write_text(replace_block(text, findings_block(Macros(load_macros()))), encoding="utf-8")
    print(f"wrote the findings block of {README.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
