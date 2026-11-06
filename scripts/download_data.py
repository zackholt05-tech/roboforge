"""Download official robomimic PH datasets listed in data/manifests (hash-verified).

Usage::

    python scripts/download_data.py                    # default set: Lift + Can, low_dim + raw
    python scripts/download_data.py --tasks lift --types low_dim
    python scripts/download_data.py --dry-run          # show what would be fetched

Each file is accepted only if its SHA-256 matches the committed manifest; an existing
file with a matching hash is skipped (idempotent). Files land under
``$ROBOFORGE_DATA_ROOT/raw`` (default ``<repo>/data/raw``), which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.data.fetch import fetch_file, load_manifest, select_files
from roboforge.paths import data_root


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=None, help="lift can square (default: lift can)")
    p.add_argument("--types", nargs="+", default=None, help="low_dim raw (default: both)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    source, files = load_manifest()
    selected = select_files(files, args.tasks, args.types)
    if not selected:
        print("No manifest entries match the selection.", file=sys.stderr)
        return 1
    print(f"data root: {data_root()}")
    records = [fetch_file(f, source, dry_run=args.dry_run) for f in selected]
    for r in records:
        print(f"[{r['action']}] {r['task']}/{r['hdf5_type']} -> {r['local_path']}")
    print(json.dumps({"n_files": len(records)}, indent=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
