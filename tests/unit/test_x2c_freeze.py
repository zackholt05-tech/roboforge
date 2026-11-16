"""The X2 freeze document and its loader (DECISIONS D-130).

- The generated document parses back to its block; the embedded test configuration is a
  byte-identical copy of the committed file, and any tampering is refused.
- The block fixes the preregistered design: the held-out suite, R = 1, 500 steps, seeds 0-2,
  the analysis design of :mod:`roboforge.x2.analysis`.
- The 12 jobs come from the block only, diffusion first, with the frozen evaluation IDs.
- Inside the container the freeze is accepted only with a clean host tree and the host's blob
  ID; on the host an uncommitted file is refused. Pinned-file changes are listed.
No held-out start is read or stepped: only the open manifest (hashes) is used.
"""

from __future__ import annotations

import json

import pytest

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.paths import REPO_ROOT
from roboforge.provenance import git_blob_sha1, sha256_file
from roboforge.sim import eval_states as es
from roboforge.x2 import analysis as xa
from roboforge.x2 import freeze_doc as fd
from roboforge.x2 import heldout as xh
from roboforge.x2 import states as x2s

PLATFORM = {"platform_id": "linux-aarch64-container-1f685638f28c",
            "image_id": "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c"}


def _document() -> tuple[str, dict, str]:
    tc_text, tc = fd.load_test_configuration()
    manifest = es.read_manifest(x2s.suite_task(), x2s.HELDOUT_SUITE)
    block = fd.build_block(tc, manifest, PLATFORM, "d" * 64, {"code": "abc1234"},
                           {"src/roboforge/x2/heldout.py": "e" * 64})
    return fd.render(block, tc_text, "2026-11-16 (UTC)"), block, tc_text


@pytest.fixture(scope="module")
def doc() -> tuple[str, dict, str]:
    if not es.manifest_path(x2s.suite_task(), x2s.HELDOUT_SUITE).exists():
        pytest.skip("X2 suites not generated")
    return _document()


def test_document_parses_back_to_its_block(doc) -> None:
    text, block, tc_text = doc
    parsed = xh.parse_x2_freeze(text)
    assert parsed == block
    assert parsed["analysis"] == xa.design()
    assert xh.embedded_test_configuration(text) == tc_text
    assert sha256_file(REPO_ROOT / xh.TEST_CONFIGURATION) == xh.TEST_CONFIGURATION_SHA256
    assert tuple(parsed["families"]) == FAMILY_ORDER
    s = parsed["suite"]
    assert (s["name"], s["n_states"], s["partition"]) == (x2s.HELDOUT_SUITE, 100, "eval_adhoc")
    assert s["manifest_content_sha256"] == es.read_manifest("can", x2s.HELDOUT_SUITE)[
        "content_sha256"]
    assert parsed["rollouts"]["horizon"] == 500 and parsed["rollouts"]["repeats"] == 1


def test_block_copies_the_twelve_checkpoints_exactly(doc) -> None:
    _, block, tc_text = doc
    tc = json.loads(tc_text)["families"]
    for fam in FAMILY_ORDER:
        want = sorted((int(r["seed"]), r["train_run_id"], int(r["selected_step"]),
                       r["checkpoint_sha256"], int(r["validation_successes"]))
                      for r in tc[fam]["runs"])
        got = [(r["seed"], r["train_run_id"], r["step"], r["checkpoint_sha256"],
                r["validation_successes"]) for r in block["families"][fam]["runs"]]
        assert got == want


@pytest.mark.parametrize("tamper", [
    lambda t: t.replace('"validation_successes": 48', '"validation_successes": 49', 1),
    lambda t: t.replace("checkpoint_sha256: 1301c3b2", "checkpoint_sha256: 1301c3b3", 1),
    lambda t: t.replace("horizon: 500", "horizon: 400", 1),
    lambda t: t.replace(f"name: {x2s.HELDOUT_SUITE}", f"name: {x2s.VALIDATION_SUITE}", 1),
    lambda t: t.replace(xh.BEGIN, "", 1),
    lambda t: t + xh.TC_BEGIN,
])
def test_tampering_is_refused(doc, tamper) -> None:
    text = tamper(doc[0])
    assert text != doc[0]
    with pytest.raises((xh.X2FreezeError, KeyError)):
        xh.parse_x2_freeze(text)


def test_jobs_come_from_the_block_in_launch_order(doc) -> None:
    _, block, _ = doc
    jobs = xh.frozen_jobs(block)
    assert [j["family"] for j in jobs] == [f for f in xh.LAUNCH_ORDER for _ in range(3)]
    assert jobs[0]["family"] == "diffusion_unet"
    for j in jobs:
        assert f"__{x2s.HELDOUT_SUITE}_step{j['step']:07d}__{PLATFORM['platform_id']}" in (
            j["eval_run_id"])
        assert j["eval_run_id"].endswith("__ps0r0") == (j["family"] == "diffusion_unet")
    j = jobs[0]
    assert xh.find_job(block, j["train_run_id"], j["step"], j["checkpoint_sha256"],
                       j["family"], j["seed"]) == j
    with pytest.raises(xh.X2FreezeError):
        xh.find_job(block, j["train_run_id"], j["step"] + 5000, j["checkpoint_sha256"],
                    j["family"], j["seed"])
    with pytest.raises(xh.X2FreezeError):
        xh.find_job(block, j["train_run_id"], j["step"], j["checkpoint_sha256"], j["family"], 3)


def test_container_attestation(doc, tmp_path, monkeypatch) -> None:
    text = doc[0]
    (tmp_path / "docs" / "extensions").mkdir(parents=True)
    (tmp_path / xh.FREEZE_REL).write_text(text)
    blob = git_blob_sha1(text.encode())
    monkeypatch.setattr(xh, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("ROBOFORGE_GIT_SHA", "a" * 40)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    assert xh.in_container()
    assert xh.load_x2_freeze(expected_blob=blob) == xh.parse_x2_freeze(text)
    for bad in (None, "", "0" * 40):
        with pytest.raises(xh.X2FreezeError):
            xh.load_x2_freeze(expected_blob=bad)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "1")
    with pytest.raises(xh.X2FreezeError):
        xh.load_x2_freeze(expected_blob=blob)


def test_host_refuses_an_uncommitted_freeze(doc, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOFORGE_GIT_SHA", raising=False)
    monkeypatch.setattr(xh, "REPO_ROOT", tmp_path)
    with pytest.raises(xh.X2FreezeError):
        xh.load_x2_freeze()  # no file
    (tmp_path / "docs" / "extensions").mkdir(parents=True)
    (tmp_path / xh.FREEZE_REL).write_text(doc[0])
    with pytest.raises(xh.X2FreezeError):
        xh.load_x2_freeze()  # not a git repository: not committed


def test_code_problems_lists_changed_and_missing_files(tmp_path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(xh, "REPO_ROOT", tmp_path)
    good = sha256_file(tmp_path / "a.py")
    assert xh.code_problems({"code_sha256": {"a.py": good}}) == []
    (tmp_path / "a.py").write_text("x = 2\n")
    assert xh.code_problems({"code_sha256": {"a.py": good, "b.py": good}}) == [
        "a.py: SHA-256 changed since the X2 freeze", "b.py: missing"]


def test_pinned_files_cover_the_x2c_code() -> None:
    for rel in fd.CODE_FILES:
        assert (REPO_ROOT / rel).is_file(), rel
    for rel in ("src/roboforge/x2/heldout.py", "src/roboforge/x2/heldout_gate.py",
                "src/roboforge/x2/analysis.py", "scripts/x2c_eval.py", "scripts/x2c_run.py",
                "scripts/analyze_x2c.py", "src/roboforge/paper/sensitivity.py"):
        assert rel in fd.CODE_FILES
    assert "Makefile" not in fd.CODE_FILES  # not mounted in the container
    assert fd.DECISIONS[0] == "D-116" and fd.DECISIONS[-1] == "D-130"
