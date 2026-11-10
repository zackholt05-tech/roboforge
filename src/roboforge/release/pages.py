"""GitHub Pages bundle of the Observatory and its live verification (DECISIONS D-102).

`stage` copies a verified local build (`website/dist`) into a fresh directory and adds:
- `.nojekyll`, so GitHub Pages serves the files as they are;
- `THIRD_PARTY_LICENSES.txt`: robosuite's MIT license (it covers the Panda, cube and can
  meshes the build extracted into `assets/`, D-090), and the license file of every npm package
  that the lockfile marks as a runtime (non-dev) dependency, including the SIL Open Font
  License texts of the bundled fonts;
- `deploy.json`: the source commit, its dirty flag, and the SHA-256 and size of every file.

`verify` fetches every file listed in `deploy.json` from the live site and compares hashes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file

WEBSITE = REPO_ROOT / "website"
DIST = WEBSITE / "dist"
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "license", "LICENCE", "OFL.txt")
# Packages that declare a license but ship no license file: the upstream text at the same
# version, committed with its source URL.
FALLBACK_LICENSES = {"@react-three/fiber": WEBSITE / "licenses" / "react-three-fiber.LICENSE"}


def _robosuite_license() -> tuple[str, str]:
    dist = importlib.metadata.distribution("robosuite")
    files = [f for f in dist.files or [] if Path(str(f)).name == "LICENSE"]
    if len(files) != 1:
        raise FileNotFoundError("robosuite LICENSE not found in the installed distribution")
    return dist.version, Path(str(dist.locate_file(files[0]))).read_text(encoding="utf-8")


def runtime_npm_packages(website: Path = WEBSITE) -> list[tuple[str, str, Path]]:
    """(name, version, directory) of every non-dev package in the lockfile."""
    lock = json.loads((website / "package-lock.json").read_text())
    out = []
    for key, meta in sorted(lock["packages"].items()):
        if not key or meta.get("dev"):
            continue
        name = key.split("node_modules/")[-1]
        out.append((name, meta["version"], website / key))
    return out


def licenses_text(website: Path = WEBSITE) -> str:
    """The THIRD_PARTY_LICENSES.txt shipped with the hosted site."""
    rs_version, rs_license = _robosuite_license()
    parts = [
        "RoboForge Observatory: third-party licenses of the files served by this site.",
        "RoboForge itself is Apache-2.0 (https://github.com/zackholt05-tech/roboforge).",
        "",
        "=" * 78,
        f"robosuite {rs_version} (MIT). The robot and object meshes in assets/ were extracted",
        "from the compiled MuJoCo model of this robosuite release at build time.",
        "=" * 78,
        rs_license.strip(),
        "",
    ]
    for name, version, path in runtime_npm_packages(website):
        lic = next((path / n for n in LICENSE_NAMES if (path / n).is_file()), None)
        lic = lic or FALLBACK_LICENSES.get(name)
        meta = json.loads((path / "package.json").read_text())
        declared = meta.get("license", "unknown")
        parts += ["=" * 78, f"{name} {version} ({declared}), npm", "=" * 78]
        if lic is None:
            raise FileNotFoundError(f"{name}: no license file in {path}")
        parts += [lic.read_text(encoding="utf-8").strip(), ""]
    return "\n".join(parts) + "\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stage(out: Path, git: dict[str, Any], dist: Path = DIST) -> dict[str, Any]:
    """Copy `dist` into a fresh `out` with the license file, `.nojekyll` and `deploy.json`."""
    if not (dist / "index.html").is_file():
        raise FileNotFoundError(f"{dist} has no index.html (run make web-build)")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dist, out)
    (out / ".nojekyll").write_text("")
    (out / "THIRD_PARTY_LICENSES.txt").write_text(licenses_text(), encoding="utf-8")
    files = {
        p.relative_to(out).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "deploy.json"
    }
    manifest = {
        "schema": "roboforge.pages_deploy/1",
        "decision": "D-102",
        "source": git,
        "files": files,
    }
    (out / "deploy.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify(base_url: str, manifest: dict[str, Any], timeout: float = 30.0) -> list[str]:
    """Fetch every staged file from the live site; problems if any hash differs."""
    base = base_url.rstrip("/") + "/"
    problems = []
    for rel, ref in manifest["files"].items():
        if rel == ".nojekyll":
            continue  # GitHub Pages does not serve dotfiles
        req = urllib.request.Request(base + rel, headers={"User-Agent": "roboforge-pages/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none hidden
            problems.append(f"{rel}: {exc}")
            continue
        if _sha(body) != ref["sha256"]:
            problems.append(f"{rel}: live SHA-256 {_sha(body)[:12]} != staged {ref['sha256'][:12]}")
    return problems
