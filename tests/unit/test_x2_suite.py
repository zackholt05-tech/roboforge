"""X2-b plan, pilot mapping and selection from files (D-125 to D-127).

* The plan resolves with the frozen suite loader to Level 1's trials, seed 0, seeds 1-2.
* Each X2-a pilot run is the seed-0 run of the trial whose configuration it has (D-125).
* Synthetic manifests and summaries go through the frozen §7 and §6.3 rules to the replicate
  runs and the X2 test configuration (the KI-062 lesson: exercise the choice step before any
  real run).
* The host guard waits for a fresh sample after every launch.
No simulator, no data; synthetic records only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.train.suite import load_plan
from roboforge.x2 import host
from roboforge.x2 import states as x2s
from roboforge.x2 import suite as xs
from roboforge.x2.pilot import FAMILIES, pilot_runs

PILOT_TRIAL = {"diffusion_unet": 2, "bc_rnn": 2, "bc_mlp_chunk": 4, "bc_mlp": 2}


def test_plan_is_level1_search_space_at_seed0() -> None:
    plan = xs.x2b_plan()
    assert (plan.task, plan.prefix, plan.tuning_seed, plan.replicate_seeds) == (
        "can", "s16x2", 0, (1, 2))
    assert tuple(f.name for f in plan.families) == FAMILIES
    level1 = load_plan(Path("configs/experiments/level1_core_can.yaml"))
    for f in plan.families:
        for t in xs.TRIALS:
            assert plan.spec(f.name, t, 0).config.sha256() == \
                level1.spec(f.name, t, 0).config.sha256()


def test_pilot_runs_are_their_trials_seed0_runs() -> None:
    plan = xs.x2b_plan()
    by_family = {r.family: r for r in pilot_runs()}
    for r in xs.tuning_runs(plan):
        if r.pilot:
            assert r.trial == PILOT_TRIAL[r.family]
            assert r.run_id == by_family[r.family].run_id
        else:
            assert r.run_id.startswith("s16x2_")
    assert sum(r.pilot for r in xs.tuning_runs(plan)) == 4
    for fam in FAMILIES:
        reps = xs.replicate_runs(plan, fam, PILOT_TRIAL[fam])
        assert [r.seed for r in reps] == [1, 2] and not any(r.pilot for r in reps)
        assert all(r.run_id.startswith("s16x2_") for r in reps)


def test_config_admits_only_the_validation_suite() -> None:
    cfg = xs.x2b_config()
    assert cfg["suite"] == x2s.VALIDATION_SUITE and cfg["n_states"] == 50
    assert cfg["summary_dirs"] == ["reports/session15/x2/pilot/evals",
                                   "reports/session16/x2/suite/evals"]


def _manifest(run: xs.PlannedRun, status: str = "complete") -> dict:
    ck = [{"step": s, "sha256": f"{run.run_id}-{s}", "path": f"x/{s}.pt"}
          for s in range(5000, 100001, 5000)]
    return {"run_id": run.run_id, "status": status, "config_sha256": run.config_sha256,
            "config": {"task": "can", "seed": run.seed,
                       "train": {"max_steps": 100000, "batch_size": 100, "ckpt_every": 5000}},
            "checkpoints": ck}


def _summary(run: xs.PlannedRun, step: int, successes: int) -> dict:
    return {"suite": x2s.VALIDATION_SUITE, "train_run_id": run.run_id,
            "eval_run_id": f"{run.run_id}__{step}", "eligible_for_selection": True,
            "n_episodes": 50, "partial": False, "successes": successes,
            "checkpoint": {"step": step, "sha256": f"{run.run_id}-{step}"},
            "first_success_step": {"mean": 200.0 if successes else None},
            "wilson95": [0.0, 1.0], "platform": {"platform_id": "canonical"}}


@pytest.fixture()
def fake_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(xs, "RUNS", tmp_path / "runs")
    sdir = tmp_path / "summaries"
    sdir.mkdir()

    def write(run: xs.PlannedRun, best_step: int, best: int) -> None:
        run.run_dir.mkdir(parents=True)
        (run.run_dir / "manifest.json").write_text(json.dumps(_manifest(run)))
        for s in range(5000, 100001, 5000):
            n = best if s == best_step else min(best - 1, 10)
            (sdir / f"{run.run_id}__{s}.json").write_text(json.dumps(_summary(run, s, n)))

    return write, sdir


def test_synthetic_suite_reaches_the_test_configuration(fake_runs) -> None:
    write, sdir = fake_runs
    plan = xs.x2b_plan()
    winners = {"diffusion_unet": 3, "bc_rnn": 1, "bc_mlp_chunk": 4, "bc_mlp": 2}
    for r in xs.tuning_runs(plan):
        write(r, 40000, 45 if r.trial == winners[r.family] else 30 + r.trial)
    summaries = xs.load_summaries([sdir])
    assert xs.check_pilot_mapping(plan)
    choices = {}
    for fam in FAMILIES:
        ch = xs.family_choice(plan, fam, summaries)
        assert ch["chosen"]["trial"] == winners[fam] and ch["chosen"]["selected_step"] == 40000
        choices[fam] = ch
    assert xs.x2_test_configuration(plan, choices, summaries) is None  # replicates missing
    for fam in FAMILIES:
        for r in xs.replicate_runs(plan, fam, winners[fam]):
            write(r, 95000, 44)
    tc = xs.x2_test_configuration(plan, choices, xs.load_summaries([sdir]))
    for fam in FAMILIES:
        rows = tc["families"][fam]["runs"]
        assert [r["seed"] for r in rows] == [0, 1, 2]
        assert [r["selected_step"] for r in rows] == [40000, 95000, 95000]
        assert tc["families"][fam]["chosen_trial"] == winners[fam]


def test_choice_waits_for_every_trial_and_every_checkpoint(fake_runs) -> None:
    write, sdir = fake_runs
    plan = xs.x2b_plan()
    runs = [r for r in xs.tuning_runs(plan) if r.family == "bc_mlp"]
    for r in runs[:3]:
        write(r, 40000, 40)
    assert xs.family_choice(plan, "bc_mlp", xs.load_summaries([sdir])) is None
    write(runs[3], 40000, 41)
    (sdir / f"{runs[3].run_id}__100000.json").unlink()
    s = xs.load_summaries([sdir])
    assert xs.selection(runs[3], s) is None
    assert xs.family_choice(plan, "bc_mlp", s) is None


def test_duplicate_or_foreign_summaries_raise(tmp_path) -> None:
    plan = xs.x2b_plan()
    r = xs.tuning_runs(plan)[0]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.json").write_text(json.dumps(_summary(r, 5000, 1)))
    (b / "y.json").write_text(json.dumps(_summary(r, 5000, 2)))
    with pytest.raises(ValueError, match="evaluated twice"):
        xs.load_summaries([a, b])
    (b / "y.json").write_text(json.dumps({**_summary(r, 10000, 2), "suite": "validation"}))
    with pytest.raises(ValueError, match="is not"):
        xs.load_summaries([a, b])


def test_remaining_disk_estimate() -> None:
    est = xs.x2b_config()["disk_estimate_gib"]
    got = xs.remaining_disk_gib({"diffusion_unet": 2, "bc_mlp": 1}, 10)
    assert got == pytest.approx(2 * est["run"]["diffusion_unet"] + est["run"]["bc_mlp"]
                                + 10 * est["evaluation"])


def test_guard_waits_for_a_fresh_sample_after_each_launch(monkeypatch) -> None:
    clock = {"t": 100.0, "si": 0}

    def fake_sample(prev):
        clock["t"] += 61
        clock["si"] += 600
        rate = None if prev is None else (clock["si"] - prev["si"]) / (clock["t"] - prev["t"])
        return {"t": clock["t"], "si": clock["si"], "so": 0, "swap_pages_per_s": rate}

    monkeypatch.setattr(host, "host_sample", fake_sample)
    monkeypatch.setattr(host, "free_gib", lambda: 50.0)
    monkeypatch.setattr(host, "tree_clean", lambda: True)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["t"])
    g = host.HostGuard(xs.x2b_config()["limits"])
    g.sample()
    assert g.hold_reason() == "await_sample"  # baseline only, no rate yet
    g.sample()
    assert g.hold_reason() is None
    g.launched()
    assert g.hold_reason() == "await_sample"
    g.sample()
    assert g.hold_reason() is None
    clock["si"] += 61 * 5000
    g.sample()
    assert g.hold_reason() == "swap"
