"""The machine-readable part of ``docs/FINAL_TEST_FREEZE.md`` (DECISIONS D-067).

The freeze document carries one YAML block between the markers :data:`BEGIN` and
:data:`END` (schema ``roboforge.heldout_freeze/1``). It fixes, per task, the held-out
suite's manifest content hash and every checkpoint that runs on it (family, chosen trial,
training run, step, SHA-256), the non-learned baseline, the rollout rule, the analysis
(bootstrap, comparisons, secondary outcome), the taxonomy hash and the SHA-256 of every code
file that runs or analyses the test. :func:`load_freeze` refuses a freeze that is not
committed unmodified (:func:`roboforge.provenance.tracked_and_unmodified`), and
:func:`code_problems` lists every frozen file whose content changed since.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file, tracked_and_unmodified

FREEZE_REL = "docs/FINAL_TEST_FREEZE.md"
SCHEMA = "roboforge.heldout_freeze/1"
BEGIN = "<!-- roboforge:freeze:begin -->"
END = "<!-- roboforge:freeze:end -->"
FAMILY_ORDER = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")
BASELINE = "random_uniform"


class FreezeError(RuntimeError):
    """The freeze is missing, uncommitted, malformed, or contradicts the repository."""


def parse_freeze(text: str) -> dict[str, Any]:
    """The YAML block between the markers (inside a ```yaml fence)."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise FreezeError("the freeze must contain exactly one machine-readable block")
    body = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    if not (body.startswith("```yaml") and body.endswith("```")):
        raise FreezeError("the machine-readable block must be a ```yaml fence")
    data = yaml.safe_load(body[len("```yaml"):-3])
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise FreezeError(f"freeze schema must be {SCHEMA}")
    for task, t in data["tasks"].items():
        if tuple(t["families"]) != FAMILY_ORDER:
            raise FreezeError(f"{task}: families must be {FAMILY_ORDER} in this order")
        for fam, f in t["families"].items():
            seeds = [r["seed"] for r in f["runs"]]
            if seeds != [0, 1, 2]:
                raise FreezeError(f"{task}/{fam}: runs must be seeds 0, 1, 2 in order")
    return data


def load_freeze(path: Path | None = None, *, require_committed: bool = True) -> dict[str, Any]:
    """Parse the freeze; with ``require_committed`` (default) it must be committed unmodified."""
    rel = FREEZE_REL if path is None else Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    if require_committed:
        ok, why = tracked_and_unmodified(rel)
        if not ok:
            raise FreezeError(f"freeze not usable: {why}")
    return parse_freeze((REPO_ROOT / rel).read_text())


def code_problems(freeze: dict[str, Any]) -> list[str]:
    """Frozen code files whose SHA-256 differs now (empty list = unchanged)."""
    out = []
    for rel, sha in sorted(freeze["code_sha256"].items()):
        p = REPO_ROOT / rel
        if not p.exists():
            out.append(f"{rel}: missing")
        elif sha256_file(p) != sha:
            out.append(f"{rel}: SHA-256 changed since the freeze")
    return out


def entries(freeze: dict[str, Any], task: str) -> list[dict[str, Any]]:
    """Every held-out evaluation of ``task``, in the frozen order: the 12 checkpoints
    (families in :data:`FAMILY_ORDER`, seeds 0-2), then the baseline."""
    t = freeze["tasks"][task]
    out = [{"task": task, "family": fam, **run} for fam in FAMILY_ORDER
           for run in t["families"][fam]["runs"]]
    return [*out, {"task": task, "family": BASELINE, **t["baseline"]}]
