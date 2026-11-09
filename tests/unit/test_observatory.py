"""Observatory selection rules and findings (DECISIONS D-089): frozen evidence, copied verbatim."""

from __future__ import annotations

import json

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.viz import obs_findings as of
from roboforge.viz import obs_select as osel

HAVE_RUNS = any((REPO_ROOT / "runs").glob("s6_diffusion_unet_lr3e4_lift_seed0_2f635398__val_*"))


def _lab(sid: str, cat: str) -> dict:
    return {"state_id": sid, "category": cat, "steps": 10}


def _tax(evals: dict) -> dict:
    return {"tasks": {"lift": [{"family": f, "seed": s, "eval_run_id": f"{f}_{s}",
                                "labels": labs} for (f, s), labs in evals.items()]}}


def test_gallery_takes_the_lowest_seed_that_has_each_outcome() -> None:
    fams = {f: [_lab("s_0", "success"), _lab("s_1", "success")] for f in osel.FAMILIES}
    evals = {(f, 0): v for f, v in fams.items()}
    evals[("bc_mlp", 1)] = [_lab("s_0", "success"), _lab("s_1", "dropped")]
    evals[("bc_mlp", 2)] = [_lab("s_0", "no_grasp_near"), _lab("s_1", "success")]
    g = osel.gallery("lift", _tax(evals))
    assert g["by_family"]["bc_mlp"]["first_failure"] == {
        "family": "bc_mlp", "seed": 1, "eval_run_id": "bc_mlp_1", "state_id": "s_1",
        "category": "dropped"}
    assert g["by_family"]["bc_rnn"]["first_failure"] is None
    assert g["by_category"]["no_grasp_near"]["seed"] == 2


def test_head_to_head_cases_and_missing_case() -> None:
    evals = {("diffusion_unet", 0): [_lab("s_0", "success"), _lab("s_1", "success"),
                                     _lab("s_2", "dropped")],
             ("bc_mlp", 0): [_lab("s_0", "dropped"), _lab("s_1", "success"),
                             _lab("s_2", "dropped")]}
    h = osel.head_to_head("lift", _tax(evals))
    assert h["a_both_succeed"]["state_id"] == "s_1"
    assert h["b_only_diffusion"]["state_id"] == "s_0"
    assert h["b_only_diffusion"]["kind"] == "case study"
    assert h["c_only_bc_mlp"] is None


def test_frozen_examples_are_reproduced_exactly() -> None:
    r = json.loads((REPO_ROOT / osel.L1_RESULTS).read_text())
    t = json.loads((REPO_ROOT / osel.L1_TAXONOMY).read_text())
    for task in ("lift", "can"):
        mine = osel.gallery(task, t)
        osel.check_against_frozen(task, mine, r["tasks"][task]["examples"])
        bad = json.loads(json.dumps(r["tasks"][task]["examples"]))
        fam = next(f for f, v in bad["by_family"].items() if v["first_success"])
        bad["by_family"][fam]["first_success"]["state_id"] = "nope"
        with pytest.raises(AssertionError):
            osel.check_against_frozen(task, mine, bad)


@pytest.mark.skipif(not HAVE_RUNS, reason="frozen evaluation records (runs/) not present")
def test_selection_covers_every_rule_and_names_three_verification_clips() -> None:
    doc = osel.build_selection()
    roles = {r for c in doc["clips"] for r in c["roles"]}
    assert {"hero", "lab", "denoise:first_success"} <= roles
    assert any(r.startswith("h2h:can:") for r in roles)
    assert any(r.startswith("pixels:") for r in roles)
    assert len(doc["verification_clips"]) == 3
    parts = {c["kind"]: c["partition"] for c in doc["clips"]}
    assert parts["validation_denoising"] == "eval_validation"
    assert parts["level2_placement"] == "eval_robustness"
    assert all(c["partition"] != "eval_final_test" for c in doc["clips"]
               if c["kind"] != "level1_heldout")


@pytest.mark.skipif(not HAVE_RUNS, reason="frozen evaluation records (runs/) not present")
def test_findings_copy_the_frozen_blocks_verbatim_with_the_erratum() -> None:
    sel = osel.build_selection()
    f = of.build_findings(sel)
    r7 = json.loads((REPO_ROOT / osel.L1_RESULTS).read_text())
    r9 = json.loads((REPO_ROOT / osel.L2_RESULTS).read_text())
    for task in ("lift", "can"):
        assert f["level1"]["tasks"][task]["success"] == r7["tasks"][task]["success"]
    assert f["level2"]["placement"] == r9["placement"]
    assert f["level2"]["training_generated"] == r9["training"]  # never silently replaced
    assert f["level2"]["training_erratum"]["corrected_ms_per_step"] == 42.2
    assert "KI-049" in f["notes"]["ki_049"]
    paths = {s["path"] for s in f["sources"]}
    assert of.L2_ERRATA in paths and osel.L1_RESULTS in paths
    states = f["level2"]["placement_map"]["states"]
    assert {s["suite"] for s in states} == {"robustness_base", "ood_cube_shift"}
    assert len(states) == 100
