"""Hash-pinned frozen sources of the paper (DECISIONS D-094).

Every table, figure and number of the paper is computed from the files below and nothing
else. Each file is pinned by SHA-256; a mismatch stops the build, so a paper can never be
generated from an edited or regenerated result file. The Observatory's ``findings.json`` is a
convenience copy and is deliberately absent, and nothing under the gitignored ``runs/`` is
read, so a fresh clone regenerates the same outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from roboforge.paths import REPO_ROOT


@dataclass(frozen=True)
class Source:
    """One frozen input: repository-relative path, pinned SHA-256 and what it is."""

    key: str
    path: str
    sha256: str
    what: str


_S = [
    # Level 1 held-out test, run once (Session 7, freeze bf0a0fa, results 97cc48a).
    Source("l1_results", "reports/session7/results/results.json",
           "b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c",
           "Level 1 held-out analysis (frozen, run once)"),
    Source("l1_taxonomy", "reports/session7/results/taxonomy.json",
           "7a71518171c8303cc00e357626ba66c64902b3d0e179ce82ba41504c2d6516b8",
           "Level 1 held-out failure taxonomy per episode"),
    Source("l1_tables", "reports/session7/results/tables.md",
           "3ea0d00dc0ec6126225d09b4026f356793d3c373e8a2ed695c8cfc072d5772b3",
           "Level 1 held-out generated tables (cross-check only)"),
    # Level 1 robustness (H4) and ablations A1-A3, validation regime (Session 7).
    Source("l1_robustness", "reports/session7/robustness/analysis.json",
           "8cca66a8a8a8cecf0b734355167936e81983c821c625a23456523365409c9451",
           "Level 1 robustness and ablation analysis (validation regime)"),
    Source("l1_robustness_tables", "reports/session7/robustness/tables.md",
           "82a22b26b3530c569871a6027f443135755ebf3d015042cbf17b66281719e933",
           "Level 1 robustness generated tables (cross-check only)"),
    # RGB study (Session 8, freeze d9c6217, D-081, results 3b23255).
    Source("rgb_results", "reports/session8/rgb_results/results.json",
           "57663b81aafc70e9c1d850322b6bb1b894f7dad0d082976b1f546475a601390e",
           "RGB study held-out analysis (frozen, run once)"),
    Source("rgb_tables", "reports/session8/rgb_results/tables.md",
           "fbde0e92efdfbcbb207be21c96bb06c09611ee6ec143b100f5d0a6be6c745c9c",
           "RGB study generated tables (cross-check only)"),
    # Level 2 (Session 9, freeze b93c50c, D-086, results 620c541 + erratum ba9a322).
    Source("l2_results", "reports/session9/level2_results/results.json",
           "1d33c975e49df907a6543b17e0cf36d047f7b5ce11aa69a355313e146c22571e",
           "Level 2 analysis, placement (H5) and held-out (frozen, run once)"),
    Source("l2_tables", "reports/session9/level2_results/tables.md",
           "13d61148de464ca403c565b1df058332eeee173062ebf3b1173db10056604790",
           "Level 2 generated tables (cross-check only)"),
    Source("l2_errata", "reports/session9/level2_results/ERRATA.md",
           "280974a994408a6f7d3683788561f0f6c05de90f4eccf8d7b1636dd90fabca31",
           "Level 2 erratum (KNOWN_ISSUES KI-049), cited beside the row it corrects"),
    Source("l2_seed0_launch", "logs/session9/pc_train/launch_records/"
           "train_gpu_pc_20261109T130430.result.json",
           "0ea097359ba49b375e3ce53199435b771e9c3253e1e039baa62f71a003e2a737",
           "Level 2 seed 0 completing launch record (KI-049 correction)"),
    # Spend: every billing event of every paid launch.
    Source("cost_ledger", "reports/compute/cost_ledger.jsonl",
           "a4f92b9f1f4ae3708bacea7a34946b3a51705205b1fc4e4a2666096f3ba40578",
           "Machine-readable cost ledger (launch, result and billing events)"),
    # Protocol and setup facts.
    Source("l1_freeze", "docs/FINAL_TEST_FREEZE.md",
           "6a3c6874c4bc1857efe46ccb05c0a1fa68247671fc9a96f5f54ce079aa9153e9",
           "Level 1 held-out freeze (machine-readable block)"),
    Source("rgb_freeze", "docs/RGB_EXPERIMENT_FREEZE.md",
           "47e7cb8ee70743dbcc6a0295a92a7462a2309870aebb3eda19060ca0e0ac385f",
           "RGB study freeze"),
    Source("l2_freeze", "docs/LEVEL2_EXPERIMENT_FREEZE.md",
           "ffe5a03e98714465acf8655c281dea653197835f4e218cfb97a1ba6173589d81",
           "Level 2 freeze"),
    Source("data_manifest", "data/manifests/robomimic_v1.5_ph.json",
           "34987db82f9a60a0da017ca5169eeae2214fd590505786063f59b85226f89188",
           "Dataset manifest (robomimic v1.5 PH, Hugging Face revision, SHA-256)"),
    Source("split_lift", "data/splits/lift_ph.json",
           "e36d89afb6d0704aa8536e23696fe30f894243ed80030e568f9a0ba3fd3a5273",
           "Lift demonstration split"),
    Source("split_can", "data/splits/can_ph.json",
           "754967ce665fca0084d3657e8e2d511db43d35f218d517c0bd15c93c17e75115",
           "Can demonstration split"),
    Source("data_audit", "reports/data_audit.json",
           "1812ad718b506247ab10361d760652b6db1286f5758a05bfb847932695dbaa21",
           "Data audit (episode lengths, control period, action statistics)"),
    Source("param_counts", "reports/session4/param_counts.json",
           "0ec7b26a0a329737cc0b18a968f5276d7130a5f66614305985505a16532486cb",
           "Exact parameter counts of the Level 1 families"),
    Source("robustness_config", "configs/eval/robustness.yaml",
           "cfa933cac8719e329c8159d2ccfa18ba2fcd58cc412b9ef65707e3e918a11d1f",
           "Preregistered robustness sets (RESEARCH_PROTOCOL §14)"),
    Source("platform_config", "configs/eval/platform.yaml",
           "a152eb41157f4b4dc6f9f1767c79d2e32c6df9b9a351430ca1dfe7af24eb2fd5",
           "Canonical evaluation platform"),
    Source("plan_lift", "configs/experiments/level1_core_lift.yaml",
           "d3de6cbcb14bcda15023a6526c6e3ba8cafb0b37d8692c5f8b6fcba871cfad3f",
           "Level 1 Lift suite plan (budget, tuning trials)"),
    Source("plan_can", "configs/experiments/level1_core_can.yaml",
           "881b4fb7fd72a33866ad4924db3b72cbe4270a212b03a8cc958efc6ba8a89e3c",
           "Level 1 Can suite plan (budget, tuning trials)"),
    Source("plan_level2", "configs/experiments/level2_pointcloud.yaml",
           "d91a546e1774779b27a16edbd5f33b0a2d4c11e6d68c71c13e2dff7e0a3e5c9d",
           "Level 2 plan"),
    Source("cfg_diffusion", "configs/models/diffusion_unet.yaml",
           "ca181c22a4af5499a67066fe2d78c06aeff5a889212ad3cceed7265fc8228bbf",
           "Level 1 diffusion configuration (base; trials override the optimizer only)"),
    Source("cfg_pc", "configs/models/pointcloud/diffusion_dp3_lift.yaml",
           "b56f515dfd861568c27d5c89ebaa9878bb87452a634daebf776f58174bfe802a",
           "Level 2 point-cloud arm configuration"),
    Source("suite_lift", "reports/session6/suite/level1_core_lift.json",
           "45c28837c1fd507946856f8acc3d9a376da4ce86de1d5222ea0a08b82c757faa",
           "Level 1 Lift suite summary (validation selections, development numbers)"),
    Source("suite_can", "reports/session7/suite/level1_core_can.json",
           "213bb2c00de1a6f6631d431cb6114d2c7776ba75fc47cbc59c963ebd9adb8629",
           "Level 1 Can suite summary (validation selections, development numbers)"),
    Source("uv_lock", "uv.lock",
           "327eaa2ea05a4e8fa46d8ed717ae4c7f5bcfa9a4069edac31b778d6b42ac5ca9",
           "Locked Python environment (package versions)"),
]

SOURCES: dict[str, Source] = {s.key: s for s in _S}


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SourceError(RuntimeError):
    """A frozen source is missing or differs from its pinned SHA-256."""


@dataclass
class Sources:
    """Verified access to the frozen sources; records which ones each output reads.

    ``use(key)`` returns the verified bytes of a source and adds it to the current output's
    provenance; ``begin(output)`` starts a new output. Hashes are checked once per process.
    """

    root: Path = REPO_ROOT
    _cache: dict[str, bytes] = field(default_factory=dict)
    _current: str | None = None
    used: dict[str, set[str]] = field(default_factory=dict)
    chained: dict[str, dict[str, str]] = field(default_factory=dict)

    def verify_all(self) -> list[str]:
        """Return a problem string for every source that is missing or has a wrong hash."""
        problems = []
        for s in SOURCES.values():
            p = self.root / s.path
            if not p.is_file():
                problems.append(f"{s.path}: missing")
            elif (got := sha256_file(p)) != s.sha256:
                problems.append(f"{s.path}: sha256 {got} != pinned {s.sha256}")
        return problems

    def begin(self, output: str) -> None:
        """Start recording the sources of one output (a repository-relative path)."""
        self._current = output
        self.used.setdefault(output, set())

    def raw(self, key: str) -> bytes:
        """Verified bytes of source ``key``; raises SourceError on a hash mismatch."""
        s = SOURCES[key]
        if key not in self._cache:
            data = (self.root / s.path).read_bytes()
            got = hashlib.sha256(data).hexdigest()
            if got != s.sha256:
                raise SourceError(f"{s.path}: sha256 {got} != pinned {s.sha256}")
            self._cache[key] = data
        if self._current is not None:
            self.used[self._current].add(key)
        return self._cache[key]

    def chained_json(self, parent: str, path: str, sha256: str) -> Any:
        """Parsed JSON of a file whose SHA-256 is recorded inside pinned source ``parent``.

        The parent's pin covers the recorded hash, so the chain is as strong as a direct pin.
        The file is listed under the current output's provenance as a chained source.
        """
        self.raw(parent)
        data = (self.root / path).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != sha256:
            raise SourceError(f"{path}: sha256 {got} != {sha256} recorded in "
                              f"{SOURCES[parent].path}")
        if self._current is not None:
            self.chained.setdefault(self._current, {})[path] = sha256
        return json.loads(data)

    def json(self, key: str) -> Any:
        """Parsed JSON of a verified source."""
        return json.loads(self.raw(key))

    def jsonl(self, key: str) -> list[dict[str, Any]]:
        """Parsed JSON lines of a verified source."""
        return [json.loads(line) for line in self.raw(key).decode().splitlines() if line]

    def yaml(self, key: str) -> Any:
        """Parsed YAML of a verified source."""
        return yaml.safe_load(self.raw(key))

    def text(self, key: str) -> str:
        """UTF-8 text of a verified source."""
        return self.raw(key).decode("utf-8")

    def freeze_block(self) -> dict[str, Any]:
        """The machine-readable YAML block of the Level 1 freeze (D-067 format)."""
        txt = self.text("l1_freeze")
        m = re.search(r"<!-- roboforge:freeze:begin -->\s*```yaml\n(.*?)```", txt, re.S)
        if not m:
            raise SourceError("docs/FINAL_TEST_FREEZE.md: machine-readable block not found")
        return yaml.safe_load(m.group(1))

    def lock_version(self, package: str) -> str:
        """The locked version of ``package`` in uv.lock (git packages: the commit)."""
        txt = self.text("uv_lock")
        m = re.search(r'\[\[package\]\]\nname = "' + re.escape(package) + r'"\nversion = "([^"]+)"'
                      r'\nsource = \{ ([^}]*) \}', txt)
        if not m:
            raise SourceError(f"uv.lock: package {package} not found")
        version, src = m.group(1), m.group(2)
        git = re.search(r'git = "[^"]*#([0-9a-f]{40})"', src)
        return f"{version}@{git.group(1)}" if git else version
