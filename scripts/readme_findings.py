"""Write (or check) the README's generated blocks: key findings and setup facts (D-103).

    uv run --frozen python scripts/readme_findings.py          # rewrite every generated block
    uv run --frozen python scripts/readme_findings.py --check  # fail if it is stale

The numbers come only from paper/generated/numbers.tex (verified by make paper-check).
"""

from __future__ import annotations

import argparse
import sys

from roboforge.paths import REPO_ROOT
from roboforge.release.macros import Macros, load_macros
from roboforge.release.readme import GENERATED_FILES, check_readme, render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="only compare, never write")
    a = p.parse_args(argv)
    if a.check:
        problems = check_readme()
        print("\n".join(problems) or "generated README and audit blocks match a fresh build")
        return 1 if problems else 0
    m = Macros(load_macros())
    for rel, names in GENERATED_FILES.items():
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"skipped {rel}: not written yet")
            continue
        path.write_text(render(path.read_text(encoding="utf-8"), m, names), encoding="utf-8")
        print(f"wrote the generated {', '.join(names)} block(s) of {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
