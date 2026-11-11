"""X3 evaluator IDs and the X3 analysis rules (DECISIONS D-111; preregistration §4-§5).

- The readings in ``scripts/analyze_x3.py`` equal the preregistration's text verbatim.
- The verdict and relation rules follow the preregistration's tables.
- The frozen seed-level intervals recomputed by the script equal the preregistration's §1.
- On synthetic matrices: identical families are not separated; a certain difference is
  separated under both methods; a missing family's contrasts are not computed.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.paths import REPO_ROOT
from roboforge.replication import x3, x3_jobs

SCRIPTS = REPO_ROOT / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ax():
    return _script("analyze_x3")


def _norm(s: str) -> str:
    return " ".join(s.split())


def test_readings_equal_the_preregistration(ax) -> None:
    text = _norm((REPO_ROOT / x3.PREREGISTRATION).read_text())
    strings = [*ax.READINGS["verdict"].values(), *ax.READINGS["relation"].values(),
               ax.READINGS["closing"], *ax.METHOD_PHRASE.values()]
    for s in strings:
        assert f'"{_norm(s)}"' in text, s
    assert ax.contrast_name("H2", "diffusion_unet", "bc_mlp") == "H2 diffusion − BC-MLP"
    assert ax.contrast_name("H2c", "diffusion_unet", "bc_mlp_chunk") == (
        "H2-c diffusion − BC-MLP-chunk")


@pytest.mark.parametrize(("boot", "seed", "want"), [
    ("separated (+)", "separated (+)", ("robust", "+")),
    ("separated (-)", "separated (-)", ("robust", "-")),
    ("separated (+)", "not separated", ("bootstrap only", "+")),
    ("separated (+)", "undefined (no seed variance)", ("bootstrap only", "+")),
    ("not separated", "separated (-)", ("seed-level only", "-")),
    ("not separated", "not separated", ("not separated", None)),
    ("separated (+)", "separated (-)", ("conflicting", None)),
])
def test_verdict_table(ax, boot: str, seed: str, want: tuple) -> None:
    assert ax.verdict(boot, seed) == want


@pytest.mark.parametrize(("frozen", "v", "sign", "boot", "want"), [
    ("separated (+)", "robust", "+", "separated (+)", "holds, robust"),
    ("separated (+)", "bootstrap only", "+", "separated (+)", "holds, bootstrap only"),
    ("separated (+)", "not separated", None, "not separated", "not reproduced"),
    ("separated (+)", "seed-level only", "+", "not separated", "not reproduced"),
    ("separated (+)", "robust", "-", "separated (-)", "reversed"),
    ("not separated", "not separated", None, "not separated", "unchanged, not separated"),
    ("not separated", "bootstrap only", "-", "separated (-)", "new separation"),
    ("not separated", "seed-level only", "+", "not separated", "new separation"),
    ("separated (+)", "conflicting", None, "separated (+)", "no relation"),
])
def test_relation_table(ax, frozen, v, sign, boot, want) -> None:
    assert ax.relation(frozen, v, sign, boot) == want


def test_reading_text(ax) -> None:
    r = ax.reading("H3 BC-RNN − BC-MLP", "bootstrap only", "+", "holds, bootstrap only")
    assert r.startswith("H3 BC-RNN − BC-MLP: separated (+) under the bootstrap but not")
    assert r.endswith("Separate replication; the frozen result stands as reported.")
    r = ax.reading("H2 diffusion − BC-RNN", "robust", "-", "new separation")
    assert "this replication separates (-) under both interval methods." in r


@pytest.fixture(scope="module")
def frozen(ax):
    if not (REPO_ROOT / ax.FROZEN_RESULTS[0]).exists():
        pytest.skip("frozen results not present")
    return ax.load_frozen()["success"]


def test_frozen_seed_level_intervals_match_the_preregistration(ax, frozen) -> None:
    mats = {f: np.ones((6, 100)) for f in FAMILY_ORDER}
    res = ax.analyse(mats, {f: list(range(6)) for f in FAMILY_ORDER}, frozen)
    got = {r["name"]: r["frozen"] for r in res["contrasts"]}
    want = {"H2 diffusion − BC-MLP": (18.7, (-4.5, 41.8), "separated (+)"),
            "H2 diffusion − BC-RNN": (-0.7, (-9.0, 7.6), "not separated"),
            "H2-c diffusion − BC-MLP-chunk": (-0.7, (-9.0, 7.6), "not separated"),
            "H3 BC-RNN − BC-MLP": (19.3, (-2.8, 41.5), "separated (+)")}
    for name, (d, ci, sep) in want.items():
        f = got[name]
        assert round(100 * f["difference"], 1) == d
        assert tuple(round(100 * x, 1) for x in f["seed_level_ci95"]) == ci
        assert f["bootstrap_separation"] == sep


def test_synthetic_controls(ax, frozen) -> None:
    rng = np.random.default_rng(0)
    base = (rng.random((6, 100)) < 0.8).astype(float)
    mats = {"bc_mlp": base.copy(), "bc_rnn": base.copy(), "bc_mlp_chunk": base.copy(),
            "diffusion_unet": np.ones((6, 100))}
    mats["bc_mlp"][:, :50] = 0.0  # a certain, large BC-MLP deficit
    seeds = {f: list(range(6)) for f in FAMILY_ORDER}
    res = ax.analyse(mats, seeds, frozen)
    rows = {r["name"]: r for r in res["contrasts"]}
    assert rows["H2 diffusion − BC-MLP"]["verdict"] == "robust"
    assert rows["H2 diffusion − BC-MLP"]["relation"] == "holds, robust"
    assert rows["H3 BC-RNN − BC-MLP"]["verdict"] == "robust"
    # Two families whose every seed row is the same start vector differ by exactly 0 on
    # every replicate (one shared state draw), whatever seeds are drawn.
    row = np.tile(base[0], (6, 1))
    same = ax.hierarchical_paired_bootstrap({"a": row, "b": row.copy()}, [("a", "b")],
                                            b=200, seed=1)
    assert same["differences"][0]["ci95"] == [0.0, 0.0]
    assert same["differences"][0]["separation"] == "not separated"
    assert res["bootstrap"]["draw_order"] == ["states", *FAMILY_ORDER]
    assert res["bootstrap"]["B"] == 10_000 and res["bootstrap"]["seed"] == 20261106


def test_missing_family_is_not_computed(ax, frozen) -> None:
    mats = {f: np.ones((6, 100)) for f in ("bc_mlp", "bc_rnn", "diffusion_unet")}
    seeds = {f: list(range(6)) for f in mats}
    res = ax.analyse(mats, seeds, frozen)
    rows = {r["name"]: r for r in res["contrasts"]}
    assert rows["H2-c diffusion − BC-MLP-chunk"]["status"].startswith("not computed")
    assert rows["H2 diffusion − BC-MLP"]["status"] == "computed"
    assert res["bootstrap"]["draw_order"] == ["states", "bc_mlp", "bc_rnn", "diffusion_unet"]


def test_seed_subsets_use_their_seeds(ax) -> None:
    mats = {f: np.vstack([np.zeros((3, 100)), np.ones((3, 100))]) for f in FAMILY_ORDER}
    seeds = {f: list(range(6)) for f in FAMILY_ORDER}
    lo = ax.analyse(mats, seeds, None, (0, 1, 2))
    hi = ax.analyse(mats, seeds, None, (3, 4, 5))
    assert all(v["estimate"] == 0.0 for v in lo["bootstrap"]["families"].values())
    assert all(v["estimate"] == 1.0 for v in hi["bootstrap"]["families"].values())
    assert all(v["n_seeds"] == 3 for v in hi["bootstrap"]["families"].values())


def test_eval_ids_follow_the_frozen_pattern() -> None:
    p = "linux-aarch64-container-1f685638f28c"
    assert x3.x3_eval_run_id("s14_bc_mlp_cos_can_seed3_04223e1b", 55000, p,
                             stochastic=False) == (
        f"s14_bc_mlp_cos_can_seed3_04223e1b__x3_heldout_step0055000__{p}")
    assert x3.x3_eval_run_id("s7_diffusion_unet_cos_can_seed0_99251da3", 65000, p,
                             stochastic=True, check=True) == (
        f"s7_diffusion_unet_cos_can_seed0_99251da3__val_step0065000__{p}__ps0r0__x3-check")


def test_jobs_take_frozen_seeds_from_the_freeze() -> None:
    fz = x3_jobs.frozen_checkpoints()
    assert len(fz) == 12
    assert sorted((c["family"], c["seed"]) for c in fz) == sorted(
        (f, s) for f in FAMILY_ORDER for s in (0, 1, 2))
    rec = json.loads((REPO_ROOT / "reports/session7/can_test_configuration.json").read_text())
    for c in fz:
        (s,) = [s for s in rec["families"][c["family"]]["seeds"] if s["seed"] == c["seed"]]
        assert (s["train_run_id"], int(s["selected_step"]), s["checkpoint_sha256"]) == (
            c["train_run_id"], c["step"], c["checkpoint_sha256"])


def test_new_run_ids_are_the_preregistered_ones() -> None:
    text = (REPO_ROOT / x3.PREREGISTRATION).read_text()
    ids = x3_jobs.new_run_ids()
    assert len(ids) == 12 and sorted(s for _, s, _ in ids) == [3] * 4 + [4] * 4 + [5] * 4
    assert all(f"`{rid}`" in text for _, _, rid in ids)


def test_tables_render_from_a_synthetic_result(ax, frozen) -> None:
    rng = np.random.default_rng(1)
    mats = {f: (rng.random((6, 100)) < p).astype(float)
            for f, p in zip(FAMILY_ORDER, (0.75, 0.95, 0.95, 0.95), strict=True)}
    lengths = {f: np.where(m > 0, 120.0, 400.0) for f, m in mats.items()}
    seeds = {f: list(range(6)) for f in FAMILY_ORDER}
    cats = ax.tx.taxonomy_config()["categories"]
    res = {"primary": ax.analyse(mats, seeds, frozen),
           "frozen_families": frozen["families"],
           "secondary": {"subsets": {k: ax.analyse(mats, seeds, None, s)
                                     for k, s in ax.SUBSETS.items()},
                         "censored_episode_length": ax.hierarchical_paired_bootstrap(
                             lengths, [(a, b) for _, a, b in ax.CONTRASTS], b=ax.B,
                             seed=ax.BOOTSTRAP_SEED, order=list(FAMILY_ORDER))},
           "engineering": {"families": {f: "complete" for f in FAMILY_ORDER},
                           "incomplete": {}, "failed_training_runs": [],
                           "n_errors_total": 0},
           "seeds_per_family": seeds,
           "evaluations": [{"family": f, "seed": s, "train_run_id": "r", "step": 5000,
                            "successes": 90, "wilson95": [0.8, 0.95], "n_errors": 0}
                           for f in FAMILY_ORDER for s in range(6)],
           "new_selections": [{"train_run_id": "r", "step": 5000, "successes": 49}],
           "taxonomy_categories": list(cats),
           "taxonomy": {f: dict.fromkeys(cats, 0) for f in FAMILY_ORDER}}
    md = ax.tables(res)
    assert md.count("separated") >= 4 and "Separate replication" in md
    assert all(r["reading"] in md for r in res["primary"]["contrasts"])
    json.dumps(res["primary"], allow_nan=False)  # results.json must serialise
