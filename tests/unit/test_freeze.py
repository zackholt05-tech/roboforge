"""The held-out freeze (D-067): block parsing, the container attestation of the committed
freeze, the freeze-block builder, and the refusals of the one-shot held-out evaluator."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from roboforge.eval import freeze as fz
from roboforge.eval import heldout as ho
from roboforge.paths import REPO_ROOT
from roboforge.provenance import git_blob_sha1, tracked_and_unmodified

FREEZE = "docs/FINAL_TEST_FREEZE.md"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block() -> dict:
    runs = [{"seed": s, "train_run_id": f"run_seed{s}", "step": 5000 * (s + 1),
             "checkpoint_sha256": f"{s}" * 64, "config_sha256": "c" * 64} for s in (0, 1, 2)]
    fam = {"chosen_trial": 1, "overrides": [], "runs": runs}
    return {"schema": fz.SCHEMA, "rollouts": {"repeats": 1, "policy_seed": 0, "horizon": 400},
            "tasks": {"lift": {"heldout_manifest_sha256": "e" * 64, "n_states": 50,
                               "families": {f: copy.deepcopy(fam) for f in fz.FAMILY_ORDER},
                               "baseline": {"policy": fz.BASELINE, "policy_seed": 0}}},
            "code_sha256": {}}


def _text(d: dict) -> str:
    return (f"# Freeze\n\nprose\n\n{fz.BEGIN}\n```yaml\n{yaml.safe_dump(d, sort_keys=False)}"
            f"```\n{fz.END}\n")


def test_block_round_trip_and_entry_order() -> None:
    d = fz.parse_freeze(_text(_block()))
    e = fz.entries(d, "lift")
    assert len(e) == 13 and e[-1]["family"] == fz.BASELINE
    assert [x["family"] for x in e[:12:3]] == list(fz.FAMILY_ORDER)
    assert [x["seed"] for x in e[:3]] == [0, 1, 2]


@pytest.mark.parametrize("bad", ["two_blocks", "schema", "family_order", "seed_order",
                                 "no_fence"])
def test_malformed_blocks_are_refused(bad: str) -> None:
    d = _block()
    text = _text(d)
    if bad == "two_blocks":
        text += text
    elif bad == "schema":
        d["schema"] = "other/1"
        text = _text(d)
    elif bad == "family_order":
        fams = d["tasks"]["lift"]["families"]
        d["tasks"]["lift"]["families"] = dict(reversed(list(fams.items())))
        text = _text(d)
    elif bad == "seed_order":
        r = d["tasks"]["lift"]["families"]["bc_mlp"]["runs"]
        r[1], r[2] = r[2], r[1]
        text = _text(d)
    else:
        text = text.replace("```yaml", "")
    with pytest.raises(fz.FreezeError):
        fz.parse_freeze(text)


def test_code_problems_names_changed_and_missing_files() -> None:
    d = _block()
    d["code_sha256"] = {"src/roboforge/eval/freeze.py": "0" * 64,
                        "src/roboforge/eval/no_such_file.py": "0" * 64}
    out = fz.code_problems(d)
    assert any("freeze.py: SHA-256 changed" in x for x in out)
    assert any("no_such_file.py: missing" in x for x in out)


def test_git_blob_sha1_equals_git_hash_object(tmp_path: Path) -> None:
    p = tmp_path / "f.md"
    p.write_bytes("freeze ✓\n\x00tail".encode())
    want = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert git_blob_sha1(p.read_bytes()) == want


def test_container_attestation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # tmp_path has no .git: the container case, attested through the wrapper's variables.
    (tmp_path / "docs").mkdir()
    f = tmp_path / FREEZE
    f.write_text("frozen\n")
    monkeypatch.setenv("ROBOFORGE_GIT_SHA", "a" * 40)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    monkeypatch.setenv("ROBOFORGE_FREEZE_BLOB", git_blob_sha1(f.read_bytes()))
    assert tracked_and_unmodified(FREEZE, repo=tmp_path)[0]
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "1")
    assert not tracked_and_unmodified(FREEZE, repo=tmp_path)[0]
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    f.write_text("edited after the commit\n")
    assert not tracked_and_unmodified(FREEZE, repo=tmp_path)[0]
    monkeypatch.setenv("ROBOFORGE_FREEZE_BLOB", "")
    assert not tracked_and_unmodified(FREEZE, repo=tmp_path)[0]
    (tmp_path / "other.md").write_text("x")
    assert not tracked_and_unmodified("other.md", repo=tmp_path)[0]


def test_host_check_uses_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOFORGE_GIT_SHA", raising=False)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "docs").mkdir()
    f = tmp_path / FREEZE
    f.write_text("v1\n")
    assert not tracked_and_unmodified(FREEZE, repo=tmp_path)[0]  # untracked
    subprocess.run([*git, "add", FREEZE], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    assert tracked_and_unmodified(FREEZE, repo=tmp_path)[0]
    f.write_text("v2\n")
    assert not tracked_and_unmodified(FREEZE, repo=tmp_path)[0]  # modified


def test_freeze_block_builder_round_trips(tmp_path: Path) -> None:
    wfb = _script("write_freeze_block")
    rec = {"task": "lift", "held_out_schedule": {"manifest_content_sha256": "e" * 64,
                                                  "n_states": 50},
           "families": {f: {"chosen_trial": 2, "overrides": ["optim.schedule=cosine"],
                            "seeds": [{"seed": s, "train_run_id": f"{f}_s{s}",
                                       "selected_step": 5000, "checkpoint_sha256": "1" * 64,
                                       "config_sha256": "2" * 64} for s in (0, 1, 2)]}
                        for f in fz.FAMILY_ORDER}}
    p = tmp_path / "lift_test_configuration.json"
    p.write_text(json.dumps(rec))
    d = wfb.block({"lift": p})
    parsed = fz.parse_freeze(f"{fz.BEGIN}\n```yaml\n{yaml.safe_dump(d, sort_keys=False)}```\n"
                             f"{fz.END}\n")
    assert parsed == d
    assert d["analysis"]["bootstrap"]["B"] == 10_000
    assert d["analysis"]["bootstrap"]["seed"] == 20261106
    assert d["rollouts"]["repeats"] == 1
    assert "scripts/analyze_heldout.py" in d["code_sha256"]
    assert fz.code_problems(d) == []


def test_heldout_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = fz.entries(fz.parse_freeze(_text(_block())), "lift")[0]
    draft = tmp_path / "draft.md"
    draft.write_text(_text(_block()))
    # A real run needs the committed freeze (here: a path that does not exist).
    monkeypatch.setattr(fz, "FREEZE_REL", "docs/NO_SUCH_FREEZE.md")
    with pytest.raises(fz.FreezeError, match="does not exist"):
        ho.evaluate_entry(entry)
    # A real run never takes a draft; a rehearsal needs one.
    with pytest.raises(ho.HeldoutError, match="draft freeze"):
        ho.evaluate_entry(entry, draft_freeze=draft)
    with pytest.raises(ho.HeldoutError, match="draft freeze"):
        ho.evaluate_entry(entry, rehearsal_suite="eval_validation")
    # A rehearsal never touches the locked suite.
    from roboforge.sim import eval_states as es
    with pytest.raises(ho.HeldoutError, match="locked"):
        ho.evaluate_entry(entry, rehearsal_suite=es.LOCKED_SUITE, draft_freeze=draft)
    # Only frozen entries run.
    other = {**entry, "step": 123}
    with pytest.raises(ho.HeldoutError, match="not an entry"):
        ho.evaluate_entry(other, rehearsal_suite="eval_validation", draft_freeze=draft)


def test_heldout_eval_ids() -> None:
    e = fz.entries(fz.parse_freeze(_text(_block())), "lift")
    assert ho.eval_id(e[0], "P", rehearsal=False) == "run_seed0__heldout_step0005000__P"
    assert ho.eval_id(e[9], "P", rehearsal=True).endswith("__rehearsal_step0005000__P__ps0r0")
    assert ho.eval_id(e[12], "P", rehearsal=False) == "random_uniform_ps0_lift__heldout__P"
