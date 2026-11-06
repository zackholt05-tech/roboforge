"""Deterministic episode-level train/validation split manifests (DECISIONS D-019).

The canonical split uses robomimic's upstream ``mask/train`` and ``mask/valid`` lists. They
are stored in the dataset file, disjoint, identical in the ``low_dim`` and ``raw`` files,
and used by prior work. The manifest records every episode ID, sorted numerically,
together with the source file hash and a hash of its own canonical content. It contains no
timestamps or paths, so rebuilding it from the same file gives byte-identical output.

There is no demonstration *test* split. Final-test evaluation uses held-out simulator
initial states (Session 2, RESEARCH_PROTOCOL §5). Demonstrations in no mask, or in both
masks, make building fail. Episodes flagged by the audit are listed under
``flagged`` with a reason. They are never removed silently.
"""

from __future__ import annotations

from typing import Any

import h5py

from roboforge.data.inspect import demo_keys
from roboforge.provenance import sha256_json

SPLIT_SCHEMA = "roboforge.split/1"


def _demo_index(name: str) -> int:
    return int(name.split("_")[-1])


def _mask(f: h5py.File, name: str) -> list[str]:
    if "mask" not in f or name not in f["mask"]:
        raise KeyError(f"dataset has no mask/{name}")
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["mask"][name][()]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"mask/{name} contains duplicate episode IDs")
    return sorted(ids, key=_demo_index)


def upstream_split(f: h5py.File, task: str, source_sha256: str, train_mask: str = "train",
                   valid_mask: str = "valid",
                   flagged: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the split manifest from upstream masks; raise if they do not partition the demos."""
    train, valid = _mask(f, train_mask), _mask(f, valid_mask)
    demos = demo_keys(f)
    overlap = set(train) & set(valid)
    if overlap:
        raise ValueError(f"train/valid masks overlap: {sorted(overlap, key=_demo_index)}")
    unknown = (set(train) | set(valid)) - set(demos)
    if unknown:
        raise ValueError(f"masks reference missing demos: {sorted(unknown, key=_demo_index)}")
    unassigned = set(demos) - set(train) - set(valid)
    if unassigned:
        raise ValueError(f"demos in no split: {sorted(unassigned, key=_demo_index)}")
    flagged = dict(sorted((flagged or {}).items(), key=lambda kv: _demo_index(kv[0])))
    lengths = {d: int(f["data"][d].attrs["num_samples"]) for d in demos}
    body = {
        "schema": SPLIT_SCHEMA,
        "task": task,
        "method": f"robomimic upstream masks mask/{train_mask} + mask/{valid_mask}",
        "source_low_dim_sha256": source_sha256,
        "splits": {"train": train, "valid": valid},
        "counts": {"train": len(train), "valid": len(valid), "total": len(demos)},
        "samples": {"train": sum(lengths[d] for d in train),
                    "valid": sum(lengths[d] for d in valid)},
        "flagged": flagged,
        "notes": "No demonstration test split: final tests use held-out simulator initial "
                 "states (Session 2). Validation demos give offline losses only; model "
                 "selection uses closed-loop validation rollouts (RESEARCH_PROTOCOL §4, §6).",
    }
    return {**body, "content_sha256": sha256_json(body)}


def verify_split(manifest: dict[str, Any]) -> None:
    """Re-check a loaded manifest's self-hash and disjointness; raise on any inconsistency."""
    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if sha256_json(body) != manifest.get("content_sha256"):
        raise ValueError("split manifest content hash mismatch (file edited or corrupted)")
    train, valid = manifest["splits"]["train"], manifest["splits"]["valid"]
    if set(train) & set(valid):
        raise ValueError("split manifest train/valid overlap")
