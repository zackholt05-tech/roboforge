"""Held-out analysis helpers on synthetic summaries (scripts/analyze_heldout.py, D-067):
pairing by state, engineering status, censoring, and the frozen example rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from roboforge.eval import freeze as fz
from roboforge.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("analyze_heldout",
                                              REPO_ROOT / "scripts" / "analyze_heldout.py")
ah = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ah)


def _summary(fam: str, succ: list[bool], *, ids=None, errors: int = 0, audit_ok: bool = True):
    ids = ids or [f"s{i:03d}" for i in range(len(succ))]
    eps = [{"state_id": i, "success": s, "first_success_step": 100 + n if s else None}
           for n, (i, s) in enumerate(zip(ids, succ, strict=True))]
    return {"family": fam}, {"episodes": eps[::-1], "n_episodes": len(eps), "n_errors": errors,
                             "verify_passed": audit_ok, "canonical_platform": True}, Path(fam)


def test_matrices_pair_by_state_and_censor_failures() -> None:
    loaded = [_summary(f, [True, False, True]) for f in (*fz.FAMILY_ORDER, fz.BASELINE)]
    succ, length, states = ah._matrices(loaded, 400)
    assert states == ["s000", "s001", "s002"]  # sorted, though summaries list them reversed
    assert succ["bc_mlp"].tolist() == [[1.0, 0.0, 1.0]]
    assert length["bc_mlp"].tolist() == [[100.0, 400.0, 102.0]]


def test_matrices_refuse_mismatched_states() -> None:
    loaded = [_summary("bc_mlp", [True, False]),
              _summary("bc_rnn", [True, False], ids=["s000", "other"])]
    with pytest.raises(ValueError, match="pairing"):
        ah._matrices(loaded, 400)


def test_engineering_status() -> None:
    ok = [_summary("bc_mlp", [True] * 50)]
    assert ah._engineering(ok, 50, False)["complete"]
    with_err = [_summary("bc_mlp", [True] * 50, errors=2)]  # in-episode errors are outcomes
    assert ah._engineering(with_err, 50, False)["complete"]
    short = [_summary("bc_mlp", [True] * 49)]
    assert not ah._engineering(short, 50, False)["complete"]
    audit = [_summary("bc_mlp", [True] * 50, audit_ok=False)]
    assert not ah._engineering(audit, 50, False)["complete"]
    missing = [({"family": "bc_mlp"}, None, REPO_ROOT / "reports" / "x.json")]
    assert ah._engineering(missing, 50, False)["missing_summaries"] == ["reports/x.json"]


def test_example_rule() -> None:
    lab = lambda pairs: [{"state_id": s, "category": c} for s, c in pairs]  # noqa: E731
    labels = {("bc_mlp", 0, "e0"): lab([("s1", "dropped"), ("s0", "dropped")]),
              ("bc_mlp", 1, "e1"): lab([("s0", "success"), ("s1", "no_grasp_near")]),
              ("bc_rnn", 0, "e2"): lab([("s0", "success"), ("s1", "no_grasp_near")])}
    out = ah._examples(labels, ["success", "dropped", "no_grasp_near", "timeout_holding"])
    fam = out["by_family"]["bc_mlp"]
    assert fam["first_success"]["seed"] == 1 and fam["first_success"]["state_id"] == "s0"
    assert fam["first_failure"]["seed"] == 0 and fam["first_failure"]["state_id"] == "s0"
    assert out["by_category"]["no_grasp_near"]["family"] == "bc_mlp"
    assert out["by_category"]["timeout_holding"] is None
    assert out["by_family"]["diffusion_unet"] == {"first_success": None, "first_failure": None}


def test_an_unavailable_family_is_left_out_and_its_comparisons_not_computed() -> None:
    import numpy as np
    frozen = {"analysis": {"bootstrap": {"B": 200, "seed": 20261106, "draw_order": [
        "states", *fz.FAMILY_ORDER, fz.BASELINE]}, "comparisons": {
        "H1": [[f, fz.BASELINE] for f in fz.FAMILY_ORDER],
        "H2": [["diffusion_unet", "bc_mlp"], ["diffusion_unet", "bc_rnn"]],
        "H2c": [["diffusion_unet", "bc_mlp_chunk"]], "H3": [["bc_rnn", "bc_mlp"]]}}}
    avail = [f for f in (*fz.FAMILY_ORDER, fz.BASELINE) if f != "bc_rnn"]
    mats = {f: np.ones((1 if f == fz.BASELINE else 3, 10)) for f in avail}
    r = ah._bootstrap(frozen, mats, ["H1", "H2", "H2c", "H3"], avail)
    assert r["draw_order"] == ["states", "bc_mlp", "bc_mlp_chunk", "diffusion_unet",
                               fz.BASELINE]
    assert {(d["a"], d["b"]) for d in r["not_computed"]} == {
        ("bc_rnn", fz.BASELINE), ("diffusion_unet", "bc_rnn"), ("bc_rnn", "bc_mlp")}
    assert len(r["differences"]) == 5 and all("hypothesis" in d for d in r["differences"])


def test_structural_issues_make_a_family_unavailable() -> None:
    loaded = [_summary("bc_mlp", [True] * 50), _summary("bc_rnn", [True] * 49)]
    eng = ah._engineering(loaded, 50, False)
    assert eng["unavailable_families"] == ["bc_rnn"] and not eng["complete"]
