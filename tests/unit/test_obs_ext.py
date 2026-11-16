"""The Observatory's v1.1 extension selection and findings (DECISIONS D-134).

- Every selected clip satisfies its rule on the committed, hash-pinned summaries, and every
  case is either a start or a reason.
- A summary whose bytes differ from its pin is refused.
- The findings copy the three pinned results and cite their hashes.
- An extension mesh absent from, or different from, the Session 10 export is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.addendum import sources as pinned
from roboforge.viz import obs_ext


@pytest.fixture(scope="module")
def selection() -> dict:
    return obs_ext.build_selection()


def test_every_case_is_a_start_or_a_reason(selection: dict) -> None:
    assert len(selection["cases"]) == 12
    for c in selection["cases"]:
        assert (c["state_id"] is None) == ("reason" in c)


def test_selected_starts_satisfy_their_rules(selection: dict) -> None:
    clips = {c["id"]: c for c in selection["clips"]}
    for case in selection["cases"]:
        if case["state_id"] is None:
            continue
        got = [clips[i] for i in case["clips"]]
        assert {c["state_id"] for c in got} == {case["state_id"]}
        ok = {c["arm"]: c["success"] for c in got}
        rule = case["role"].split(":")[-1]
        if rule == "a_only_diffusion":
            assert ok["diffusion_unet"] and not any(v for k, v in ok.items()
                                                    if k != "diffusion_unet")
        elif rule.startswith("b_only_"):
            assert not ok["diffusion_unet"] and ok[rule.removeprefix("b_only_")]
        elif rule in ("c_both_succeed", "b_all_succeed"):
            assert all(ok.values())
        elif rule == "c_clamp_changed_outcome":
            assert ok["bc_mlp_chunk"] != ok["bc_mlp_chunk+clamp"]


def test_rules_pick_the_first_matching_start(selection: dict) -> None:
    strips = selection["strips"]["x2"]
    order = strips["diffusion_unet"]["order"]
    d, b = strips["diffusion_unet"]["outcomes"], strips["bc_rnn"]["outcomes"]
    first = next(order[i] for i in range(len(order)) if d[i] and not b[i])
    case = next(c for c in selection["cases"] if c["role"] == "x2:bc_rnn:a_only_diffusion")
    assert case["state_id"] == first


def test_x2_clips_carry_the_study_label(selection: dict) -> None:
    for c in selection["clips"]:
        assert (c["display_label"] == obs_ext.X2_LABEL) == (c["study"] == "x2")
    assert all(v in {c["id"] for c in selection["clips"]}
               for v in selection["verification_clips"])
    assert len(selection["verification_clips"]) == 2


def test_a_changed_summary_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"episodes": []}))
    with pytest.raises(RuntimeError, match="differs from the pinned"):
        obs_ext.read_pinned("s.json", "0" * 64, root=tmp_path)


def test_findings_copy_the_pinned_results() -> None:
    f = obs_ext.build_findings()
    assert [(s["path"], s["sha256"]) for s in f["sources"]] == [
        (s.path, s.sha256) for s in pinned.SOURCES.values()]
    x2 = pinned.load("x2")
    assert f["x2"]["contrasts"] == x2["primary"]["contrasts"]
    assert f["x3"]["contrasts"] == pinned.load("x3")["primary"]["contrasts"]
    assert f["x1"]["readings"] == pinned.load("x1")["readings"]


def test_mesh_check_refuses_new_or_changed_meshes(tmp_path: Path) -> None:
    import importlib.util
    import sys

    from roboforge.paths import REPO_ROOT
    spec = importlib.util.spec_from_file_location(
        "export_observatory_ext", REPO_ROOT / "scripts" / "export_observatory_ext.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_observatory_ext"] = mod
    spec.loader.exec_module(mod)
    s10 = json.loads((REPO_ROOT / mod.SESSION10 / "meshes.json").read_text())["meshes"]
    name = sorted(s10)[0]
    (tmp_path / "meshes.json").write_text(json.dumps({"meshes": {name: s10[name]}}))
    assert mod.mesh_problems(tmp_path) == []
    (tmp_path / "meshes.json").write_text(json.dumps(
        {"meshes": {name: {**s10[name], "n_vert": -1}, "new_mesh": {}}}))
    assert len(mod.mesh_problems(tmp_path)) == 2
