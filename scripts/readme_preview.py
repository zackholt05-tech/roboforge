"""Build (or check) the README preview GIF from recorded head-to-head renders (DECISIONS D-103).

    uv run --frozen python scripts/readme_preview.py          # write docs/media/readme_preview.*
    uv run --frozen python scripts/readme_preview.py --check  # hashes only, no re-encode
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.release import preview


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="verify the sidecar's hashes only")
    a = p.parse_args(argv)
    if a.check:
        problems = preview.check()
        print("\n".join(problems) or "README preview matches its sources and sidecar")
        return 1 if problems else 0
    side = preview.build()
    print(json.dumps(side["output"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
