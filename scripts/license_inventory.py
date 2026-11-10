"""License inventory of the locked Python environment and the site's npm packages (D-104).

    uv run --frozen python scripts/license_inventory.py --out reports/session12/licenses.json

Python: every distribution installed in the uv environment, with its declared license
(`License-Expression`, `License` or the license classifiers). npm: every package in
`website/package-lock.json`, runtime and dev, with the `license` field of its installed
`package.json`. Packages whose license is unknown or matches a strong-copyleft family are
listed under "review"; nothing is decided automatically.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

from roboforge.paths import REPO_ROOT

COPYLEFT = re.compile(r"\bA?GPL|\bLGPL|SSPL|EUPL|CC-BY-SA|General Public License", re.I)


def python_licenses() -> list[dict[str, str]]:
    out = []
    for dist in importlib.metadata.distributions():
        md = dist.metadata
        lic = md.get("License-Expression") or ""
        if not lic:
            raw = (md.get("License") or "").strip()
            lic = raw.splitlines()[0][:80] if raw else ""
        if not lic or len(lic) > 60:
            cls = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or []
                   if c.startswith("License ::")]
            lic = "; ".join(cls) or lic or "UNKNOWN"
        out.append({"name": md["Name"], "version": dist.version, "license": lic})
    return sorted(out, key=lambda r: r["name"].lower())


def npm_licenses(website: Path = REPO_ROOT / "website") -> list[dict[str, str]]:
    lock = json.loads((website / "package-lock.json").read_text())
    out = []
    for key, meta in sorted(lock["packages"].items()):
        if not key:
            continue
        pj = website / key / "package.json"
        lic = meta.get("license")
        if lic is None and pj.is_file():
            lic = json.loads(pj.read_text()).get("license")
        out.append({"name": key.split("node_modules/")[-1], "version": meta.get("version", "?"),
                    "license": str(lic or "UNKNOWN"), "dev": bool(meta.get("dev"))})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    py, npm = python_licenses(), npm_licenses()
    review = [r | {"ecosystem": "python"} for r in py
              if r["license"] == "UNKNOWN" or COPYLEFT.search(r["license"])]
    review += [r | {"ecosystem": "npm"} for r in npm
               if r["license"] == "UNKNOWN" or COPYLEFT.search(r["license"])]
    doc = {"schema": "roboforge.license_inventory/1", "decision": "D-104", "python": py,
           "npm": npm, "review": review}
    out = a.out if a.out.is_absolute() else REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"python {len(py)} distributions, npm {len(npm)} packages; review {len(review)}:")
    for r in review:
        print(f"  {r['ecosystem']:6s} {r['name']} {r['version']}: {r['license']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
