"""The X2-c analysis on synthetic outcomes, end to end (preregistration §7; DECISIONS D-130).

Synthetic first (the KI-062 lesson): 12 synthetic summaries in a temporary directory go
through ``scripts/analyze_x2c.py``'s :func:`run`, which writes ``results.json``, ``tables.md``,
``taxonomy.json`` and ``analysis_record.json``. Covered:

- the frozen bootstrap, unchanged (B = 10,000, seed 20261106, the frozen draw order) and the
  D-095 Welch interval, including "undefined (no seed variance)";
- the verdict and relation rules, including "neither separated", "conflicting" and "no
  relation";
- the censored length with failures at 500 steps;
- an ENGINEERING-INCOMPLETE family (skipped in the draw order, its contrasts not computed);
- every summary check, a taxonomy that disagrees with its summary, and the run-once rule.
No file under ``runs/`` and no held-out state is read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.eval.statistics import hierarchical_paired_bootstrap
from roboforge.eval.taxonomy import taxonomy_config
from roboforge.paths import REPO_ROOT
from roboforge.x2 import analysis as xa
from roboforge.x2 import freeze_doc as fd
from roboforge.x2 import heldout as xh
from roboforge.x2.data import DATASET_ID

PLATFORM = {"platform_id": "linux-aarch64-container-1f685638f28c", "image_id": "sha256:x"}
BLOB = "b" * 40
STATES = [f"s{i:03d}" for i in range(100)]
RATES = {"bc_mlp": (0.30, 0.20, 0.25), "bc_rnn": (0.70, 0.60, 0.75),
         "bc_mlp_chunk": (1.0, 1.0, 1.0), "diffusion_unet": (1.0, 1.0, 1.0)}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ax():
    return _script("analyze_x2c")


@pytest.fixture(scope="module")
def block() -> dict:
    _, tc = fd.load_test_configuration()
    suite = {"n_states": 100, "content_sha256": "c" * 64, "partition": "eval_adhoc"}
    return fd.build_block(tc, suite, PLATFORM, "d" * 64, {"code": "abc1234"}, {})


@pytest.fixture(scope="module")
def frozen(ax) -> dict:
    return ax._pinned_json(*xa.FROZEN_RESULTS)["tasks"]["can"]["success"]


def _summary(j: dict, block: dict) -> dict:
    rate = RATES[j["family"]][j["seed"]]
    rng = np.random.default_rng(1000 * FAMILY_ORDER.index(j["family"]) + j["seed"])
    ok = np.zeros(100, dtype=bool)
    ok[rng.choice(100, size=round(100 * rate), replace=False)] = True
    eps = [{"state_id": sid, "success": bool(s), "first_success_step": 150 + i % 50 if s
            else None, "steps": 150 + i % 50 if s else 500, "error": None}
           for i, (sid, s) in enumerate(zip(STATES, ok, strict=True))]
    return {"schema": xh.SCHEMA, "label": xh.LABEL, "study": xh.STUDY,
            "eval_run_id": j["eval_run_id"], "canonical_platform": True,
            "platform": {"platform_id": PLATFORM["platform_id"]}, "partial": False,
            "n_episodes": 100, "episodes": eps, "verify_passed": True,
            "suite": block["suite"]["name"],
            "suite_manifest_content_sha256": block["suite"]["manifest_content_sha256"],
            "checkpoint": {"step": j["step"], "sha256": j["checkpoint_sha256"]},
            "train_run_id": j["train_run_id"], "family": j["family"], "seed": j["seed"],
            "dataset_id": DATASET_ID, "episode_rules": {"horizon": 500},
            "git": {"sha": "f" * 40, "dirty": False}, "freeze": {"git_blob": BLOB},
            "attempt": 1, "rerun_decision": None, "successes": int(ok.sum()), "n_errors": 0,
            "wall_s": 100.0, "records_dir": f"runs/x2c/{j['eval_run_id']}"}


def _write(tmp: Path, block: dict, skip: set[str] = frozenset()) -> tuple[Path, dict]:
    d = tmp / "evals"
    d.mkdir(parents=True)
    out = {}
    for j in xh.frozen_jobs(block):
        s = _summary(j, block)
        out[j["eval_run_id"]] = s
        if j["eval_run_id"] not in skip:
            (d / f"{j['eval_run_id']}.json").write_text(json.dumps(s))
    return d, out


def _classify(summaries: dict, *, lie: bool = False):
    def classify(records_dir: Path) -> list[dict]:
        s = summaries[Path(records_dir).name]
        return [{"state_id": e["state_id"],
                 "category": "success" if (e["success"] or lie) else "timeout_holding"}
                for e in s["episodes"]]
    return classify


def _run(ax, tmp: Path, block: dict, frozen: dict, skip=frozenset(), lie=False) -> dict:
    d, summaries = _write(tmp, block, skip)
    jobs = xh.frozen_jobs(block)
    latency = {"rows": [{"checkpoint_sha256": j["checkpoint_sha256"],
                         "observe_plus_act_ms": {"median": 1.5}} for j in jobs]}
    return ax.run(tmp / "out", d, jobs, block, BLOB, frozen, latency, STATES,
                  classify=_classify(summaries, lie=lie),
                  categories=list(taxonomy_config()["categories"]), taxonomy_sha256="t",
                  commit_ok=lambda sha: True, decision_ok=lambda dec: dec == "D-999")


def _by_name(res: dict) -> dict:
    return {r["name"]: r for r in res["primary"]["contrasts"]}


def test_end_to_end_on_synthetic_outcomes(ax, block, frozen, tmp_path) -> None:
    res = _run(ax, tmp_path, block, frozen)
    for name in ("results.json", "tables.md", "taxonomy.json", "analysis_record.json"):
        assert (tmp_path / "out" / name).is_file()
    boot = res["primary"]["bootstrap"]
    assert boot["draw_order"] == ["states", *FAMILY_ORDER]
    assert (boot["B"], boot["seed"]) == (10_000, 20261106)
    c = _by_name(res)
    h2 = c["H2 diffusion − BC-MLP"]
    assert (h2["verdict"], h2["sign"], h2["relation"]) == ("robust", "+", "same direction")
    h2r = c["H2 diffusion − BC-RNN"]
    assert (h2r["verdict"], h2r["relation"]) == ("robust", "new separation")
    h2c = c["H2-c diffusion − BC-MLP-chunk"]
    assert h2c["seed_level_separation"] == xa.UNDEFINED and h2c["seed_level_ci95"] is None
    assert (h2c["verdict"], h2c["relation"]) == ("not separated", "neither separated")
    h3 = c["H3 BC-RNN − BC-MLP"]
    assert (h3["verdict"], h3["relation"]) == ("robust", "same direction")
    assert h2["frozen"]["bootstrap_separation"] == "separated (+)"
    assert h2c["frozen"]["bootstrap_separation"] == "not separated"
    assert h2["frozen"]["n_states"] == 50 and h2["frozen"]["horizon"] == 400
    tables = (tmp_path / "out" / "tables.md").read_text()
    for s in ("X2 (Can MH)", "never pooled", "neither separated", "undefined",
              "within-operator", "500 steps"):
        assert s in tables, s
    assert sum(res["taxonomy"]["bc_mlp"].values()) == 300
    assert res["taxonomy"]["diffusion_unet"]["success"] == 300
    assert res["engineering"]["families"] == {f: "complete" for f in FAMILY_ORDER}
    record = json.loads((tmp_path / "out" / "analysis_record.json").read_text())
    assert len(record["inputs"]) == 15
    with pytest.raises(SystemExit):  # the analysis runs once: results.json exists
        ax.run(tmp_path / "out", tmp_path / "evals", xh.frozen_jobs(block), block, BLOB,
               frozen, {"rows": []}, STATES, classify=None, categories=[], taxonomy_sha256="t",
               commit_ok=lambda s: True, decision_ok=lambda d: False)


def test_statistics_are_the_frozen_functions_unchanged(ax, block, frozen, tmp_path) -> None:
    res = _run(ax, tmp_path, block, frozen)
    _, summaries = _write(tmp_path / "w", block)
    evals = [{"job": j, "summary": summaries[j["eval_run_id"]]} for j in xh.frozen_jobs(block)]
    succ, length = xa.matrices(evals, STATES, list(FAMILY_ORDER))
    comps = [(a, b) for _, a, b in xa.CONTRASTS]
    direct = hierarchical_paired_bootstrap(succ, comps, b=10_000, seed=20261106,
                                           order=list(FAMILY_ORDER))
    assert direct["differences"] == res["primary"]["bootstrap"]["differences"]
    cel = hierarchical_paired_bootstrap(length, comps, b=10_000, seed=20261106,
                                        order=list(FAMILY_ORDER))
    assert cel == res["primary"]["censored_episode_length"]
    assert length["bc_mlp"].max() == 500.0 and length["diffusion_unet"].max() < 500.0
    assert np.array_equal(succ["bc_mlp_chunk"], np.ones((3, 100)))


def test_an_engineering_incomplete_family(ax, block, frozen, tmp_path) -> None:
    (rnn1,) = [j for j in xh.frozen_jobs(block) if (j["family"], j["seed"]) == ("bc_rnn", 1)]
    res = _run(ax, tmp_path, block, frozen, skip={rnn1["eval_run_id"]})
    assert res["primary"]["bootstrap"]["draw_order"] == ["states", "bc_mlp", "bc_mlp_chunk",
                                                         "diffusion_unet"]
    assert res["engineering"]["families"]["bc_rnn"] == "ENGINEERING-INCOMPLETE"
    assert res["engineering"]["incomplete"][rnn1["eval_run_id"]]["problems"] == ["no summary"]
    c = _by_name(res)
    for name in ("H2 diffusion − BC-RNN", "H3 BC-RNN − BC-MLP"):
        assert c[name]["status"] == xa.STATUS_INCOMPLETE
    assert c["H2 diffusion − BC-MLP"]["status"] == "computed"
    assert "ENGINEERING-INCOMPLETE" in (tmp_path / "out" / "tables.md").read_text()


def test_a_taxonomy_that_disagrees_with_its_summary_stops(ax, block, frozen, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _run(ax, tmp_path, block, frozen, lie=True)


@pytest.mark.parametrize(("mutate", "problem"), [
    (lambda s: s["episode_rules"].update(horizon=400), "horizon"),
    (lambda s: s["git"].update(dirty=True), "dirty tree"),
    (lambda s: s["freeze"].update(git_blob="0" * 40), "freeze blob"),
    (lambda s: s["checkpoint"].update(sha256="0" * 64), "checkpoint"),
    (lambda s: s.update(n_episodes=99), "not complete"),
    (lambda s: s.update(canonical_platform=False), "platform"),
    (lambda s: s.update(verify_passed=False), "audit failed"),
    (lambda s: s.update(suite_manifest_content_sha256="0" * 64), "suite"),
    (lambda s: s.update(label="other"), "schema/label"),
    (lambda s: s.update(attempt=2), "attempt"),
    (lambda s: s.update(attempt=2, rerun_decision="D-998"), "attempt"),
    (lambda s: s.update(attempt=2, rerun_decision="D-999"), None),
])
def test_summary_checks(ax, block, mutate, problem) -> None:
    j = xh.frozen_jobs(block)[0]
    s = _summary(j, block)
    mutate(s)
    got = ax.check_summary(s, j, block, BLOB, decision_ok=lambda d: d == "D-999")
    assert got == ([] if problem is None else [problem])


def test_a_summary_from_another_commit_is_incomplete(ax, block, tmp_path) -> None:
    d, _ = _write(tmp_path, block)
    evals, incomplete = ax.load_evaluations(d, xh.frozen_jobs(block), block, BLOB,
                                            commit_ok=lambda sha: False,
                                            decision_ok=lambda dec: False)
    assert evals == [] and len(incomplete) == 12
    assert all(v["problems"] == ["commit does not hold the X2 freeze and its pinned files"]
               for v in incomplete.values())


@pytest.mark.parametrize(("boot", "seed", "want"), [
    ("separated (+)", "separated (+)", ("robust", "+")),
    ("separated (-)", "separated (-)", ("robust", "-")),
    ("separated (+)", "not separated", ("bootstrap only", "+")),
    ("separated (-)", xa.UNDEFINED, ("bootstrap only", "-")),
    ("not separated", "separated (+)", ("seed-level only", "+")),
    ("not separated", xa.UNDEFINED, ("not separated", None)),
    ("separated (+)", "separated (-)", ("conflicting", None)),
])
def test_verdict(boot, seed, want) -> None:
    assert xa.verdict(boot, seed) == want


@pytest.mark.parametrize(("frozen_sep", "x2_sep", "verdict", "want"), [
    ("separated (+)", "separated (+)", "robust", "same direction"),
    ("separated (+)", "separated (+)", "bootstrap only", "same direction"),
    ("separated (+)", "separated (-)", "robust", "reversed"),
    ("separated (+)", "not separated", "seed-level only", "not reproduced"),
    ("not separated", "separated (-)", "bootstrap only", "new separation"),
    ("not separated", "not separated", "not separated", "neither separated"),
    ("not separated", "not separated", "seed-level only", "neither separated"),
    ("separated (+)", "separated (+)", "conflicting", "no relation"),
])
def test_relation(frozen_sep, x2_sep, verdict, want) -> None:
    assert xa.relation(frozen_sep, x2_sep, verdict) == want
    assert want in xa.RELATIONS


def test_design_is_the_preregistered_one() -> None:
    text = " ".join((REPO_ROOT / xh.PREREGISTRATION).read_text().split())
    d = xa.design()
    assert "B = 10,000, seed 20261106" in text and d["bootstrap"]["B"] == 10_000
    assert d["bootstrap"]["seed"] == 20261106
    assert ("draw order states, bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet" in text
            and d["bootstrap"]["draw_order"] == ["states", *FAMILY_ORDER])
    assert ("H2 diffusion_unet − bc_mlp; H2 diffusion_unet − bc_rnn; H2c diffusion_unet − "
            "bc_mlp_chunk; H3 bc_rnn − bc_mlp") in text
    assert [(c["hypothesis"], c["a"], c["b"]) for c in d["contrasts"]] == list(xa.CONTRASTS)
    for label in ("same direction", "new separation", "not reproduced", "reversed"):
        assert f'"{label}"' in text and label in d["relation"]["labels"]
    for v in ("robust (±)", "bootstrap only (±)", "seed-level only (±)", "not separated",
              "conflicting"):
        assert v in text
    assert "first-success step, else 500" in text and xa.HORIZON == 500


def test_main_end_to_end_with_the_gate_stubbed(ax, block, tmp_path, monkeypatch) -> None:
    """``main()`` itself: the gate's results stubbed, every input and output in a temporary
    tree (the committed Level 1 result and latency file copied in, read by their pins)."""
    import shutil

    for rel in (xa.FROZEN_RESULTS[0], xa.LATENCY[0]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    _, summaries = _write(tmp_path / "reports" / "session17" / "x2" / "heldout", block)
    monkeypatch.setenv("ROBOFORGE_EVIDENCE_SESSION", "17")
    monkeypatch.setattr(ax, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ax, "repo_relative",
                        lambda p: Path(p).resolve().relative_to(tmp_path.resolve()).as_posix())
    monkeypatch.setattr(ax.xh, "load_x2_freeze", lambda **k: block)
    monkeypatch.setattr(ax.xh, "freeze_blob", lambda: BLOB)
    monkeypatch.setattr(ax.xg, "check_inputs", lambda f: {"jobs": xh.frozen_jobs(f)})
    monkeypatch.setattr(ax.xg, "existing_heldout_outputs", lambda: [])
    monkeypatch.setattr(ax.xg, "decision_recorded", lambda d: False)
    monkeypatch.setattr(ax, "commit_matches_freeze", lambda sha, f, b: True)
    monkeypatch.setattr("roboforge.x2.host.tree_clean", lambda: True)
    monkeypatch.setattr("roboforge.sim.eval_states.read_manifest",
                        lambda task, name: {"states": [{"state_id": s} for s in STATES]})
    monkeypatch.setattr("roboforge.eval.taxonomy.classify_records", _classify(summaries))
    assert block["outputs"]["analysis"] == "reports/session17/x2"
    assert ax.main([]) == 0
    out = tmp_path / "reports" / "session17" / "x2"
    res = json.loads((out / "results.json").read_text())
    assert res["freeze"]["git_blob"] == BLOB and len(res["evaluations"]) == 12
    assert res["frozen_source"]["sha256"] == xa.FROZEN_RESULTS[1]
    assert all(e["latency_ms_median"] is not None for e in res["evaluations"])
    assert res["clarifications"] == block["clarifications"]
    with pytest.raises(SystemExit):
        ax.main([])  # once


def test_commit_matches_freeze(ax, monkeypatch) -> None:
    import subprocess

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    rel = xh.PREREGISTRATION  # a committed file stands in for the freeze
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    data = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, capture_output=True,
                          check=True).stdout
    import hashlib

    pins = {"code_sha256": {rel: hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr(ax.xh, "FREEZE_REL", rel)
    assert ax.commit_matches_freeze("HEAD", pins, blob)
    assert not ax.commit_matches_freeze("HEAD", pins, "0" * 40)
    assert not ax.commit_matches_freeze("HEAD", {"code_sha256": {rel: "0" * 64}}, blob)
    assert not ax.commit_matches_freeze("0" * 40, pins, blob)


def test_decision_recorded(monkeypatch) -> None:
    from roboforge.x2 import heldout_gate as xg

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    monkeypatch.setattr(xg, "tracked_and_unmodified", lambda rel: (True, "committed"))
    assert xg.decision_recorded("D-130") and xg.decision_recorded("D-095")
    assert not xg.decision_recorded("D-999") and not xg.decision_recorded("D-13")
