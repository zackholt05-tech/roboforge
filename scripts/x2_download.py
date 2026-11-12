"""X2: download and SHA-256-verify the Can MH files of ``data/manifests/robomimic_v1.5_mh.json``.

DECISIONS D-118. A thin wrapper over the frozen :mod:`roboforge.data.fetch`: the new manifest
is loaded by name, and every file is accepted only if its size and SHA-256 equal the manifest
(a mismatch deletes the partial file and stops). Existing files with a matching hash are
skipped. The PH manifest and ``scripts/download_data.py`` are unchanged. Paths are printed
relative to the data root, so the log carries no local absolute path::

    python scripts/x2_download.py            # both files
    python scripts/x2_download.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.data.fetch import fetch_file, load_manifest, select_files
from roboforge.paths import data_root
from roboforge.x2.data import X2_MANIFEST


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    source, files = load_manifest(X2_MANIFEST)
    selected = select_files(files, None, None)
    root = data_root()
    out = []
    for f in selected:
        r = fetch_file(f, source, dry_run=args.dry_run)
        rel = Path(r["local_path"]).relative_to(root).as_posix()
        out.append({"task": r["task"], "dataset_type": f.dataset_type,
                    "hdf5_type": r["hdf5_type"], "action": r["action"], "path": rel,
                    "url": r["url"], "expected_sha256": r["expected_sha256"],
                    "sha256": r.get("sha256"), "resumed_from_bytes": r.get("resumed_from_bytes")})
    print(json.dumps({"manifest": X2_MANIFEST, "n_files": len(out), "files": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
