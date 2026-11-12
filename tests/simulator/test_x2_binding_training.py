"""The X2 binding changes only the data source of the frozen trainer and policy (D-121).

* A short training run inside the binding, with the loader pointed at the Can PH artifacts, is
  content-identical to the same run without the binding.
* A short run on the X2 (Can MH) data records the X2 dataset and hashes.
* The frozen ``LearnedPolicy`` refuses the X2 checkpoint outside the binding; inside it, the
  X2 checkpoint loads and a Can PH checkpoint is refused.
* ``RoboForgeEnv('can')`` has the X2 horizon (500) inside the binding and 400 outside.

The tiny models here are pipeline fixtures, not results. Needs the simulator and the processed
Can PH and X2 data.
"""

from __future__ import annotations

import pytest

from roboforge.data import pipeline
from roboforge.data.processed import load_processed, processed_path
from roboforge.eval.learned import LearnedPolicy
from roboforge.paths import REPO_ROOT
from roboforge.train.checkpoint import CheckpointError, checkpoint_differences
from roboforge.train.config import load_config
from roboforge.train.run import train_run
from roboforge.x2 import data as x2
from roboforge.x2.binding import x2_bindings

pytestmark = [pytest.mark.sim, pytest.mark.data]
TINY = ["train.max_steps=30", "train.ckpt_every=30", "train.val_every=30", "train.log_every=15",
        "task=can"]


def _ph_episodes(docs):
    proc = docs["task_manifest"]["processed"]
    path = processed_path("can", proc["converter_version"])
    return load_processed(path, expected_sha256=proc["content_sha256"])


@pytest.fixture(scope="module")
def data_ready():
    proc = pipeline.load_task_artifacts("can")["task_manifest"]["processed"]
    if not processed_path("can", proc["converter_version"]).is_file():
        pytest.skip("Can PH processed data not built (run `make build-data`)")
    if not x2.processed_path().is_file():
        pytest.skip("X2 processed data not built (run `make x2-data`)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, data_ready):
    cfg = load_config(REPO_ROOT / "configs" / "models" / "bc_mlp.yaml", TINY)
    out = {}
    out["plain"] = train_run(cfg, tmp_path_factory.mktemp("plain"), "test")
    frozen_loader = pipeline.load_task_artifacts
    with x2_bindings(loader=frozen_loader, episodes_of=_ph_episodes):
        out["bound_ph"] = train_run(cfg, tmp_path_factory.mktemp("bound_ph"), "test")
    with x2_bindings():
        out["x2"] = train_run(cfg, tmp_path_factory.mktemp("x2"), "test")
    return out


def _ckpt(manifest):
    c = manifest["checkpoints"][-1]
    return c["path"], c["sha256"]


def test_binding_changes_nothing_but_the_data_source(runs) -> None:
    a, b = runs["plain"], runs["bound_ph"]
    assert a["data"] == b["data"] and a["windows"] == b["windows"]
    assert checkpoint_differences(_ckpt(a)[0], _ckpt(b)[0]) == []


def test_x2_run_records_the_x2_dataset(runs) -> None:
    m, docs = runs["x2"], x2.load_x2_artifacts()
    assert m["status"] == "complete"
    assert m["data"]["dataset_id"] == "robomimic_v1.5/can/mh"
    assert m["data"]["normalization_sha256"] == docs["normalization"]["content_sha256"]
    assert m["data"]["split_sha256"] == docs["split"]["content_sha256"]
    assert m["windows"]["train_episodes"] == 270 and m["windows"]["valid_episodes"] == 30


def test_policies_load_only_under_their_own_contract(runs) -> None:
    x2_path, x2_sha = _ckpt(runs["x2"])
    ph_path, ph_sha = _ckpt(runs["plain"])
    with pytest.raises(CheckpointError, match="committed can contract"):
        LearnedPolicy.from_checkpoint(x2_path, expected_sha256=x2_sha)
    LearnedPolicy.from_checkpoint(ph_path, expected_sha256=ph_sha)
    with x2_bindings():
        LearnedPolicy.from_checkpoint(x2_path, expected_sha256=x2_sha)
        with pytest.raises(CheckpointError, match="committed can contract"):
            LearnedPolicy.from_checkpoint(ph_path, expected_sha256=ph_sha)


def test_environment_horizon_follows_the_binding(data_ready) -> None:
    from roboforge.sim.env_adapter import RoboForgeEnv

    env = RoboForgeEnv("can")
    try:
        assert env.horizon == 400
    finally:
        env.close()
    with x2_bindings():
        env = RoboForgeEnv("can")
        try:
            assert env.horizon == 500
            assert env._hashes["task_spec_sha256"] == x2.load_x2_artifacts()[
                "task_manifest"]["task_spec_sha256"]
        finally:
            env.close()


def test_x2_evaluator_runs_x2_validation_starts_with_the_x2_rules(runs, tmp_path,
                                                                   monkeypatch) -> None:
    """The evaluator on 2 X2 validation starts (tiny model, development only)."""
    import roboforge.sim.eval_states as es
    from roboforge.x2 import states as x2s
    from roboforge.x2.evaluate import evaluate_x2

    path, sha = _ckpt(runs["x2"])
    run_dir = runs["x2"]["checkpoints"][-1]["path"].rsplit("/checkpoints/", 1)[0]
    with pytest.raises(RuntimeError, match="inside x2_bindings"):
        evaluate_x2(run_dir, 30, family="bc_mlp", expected_sha256=sha, runs_dir=tmp_path)
    real = es.load_suite
    monkeypatch.setattr(es, "load_suite", lambda task, suite: real(task, suite)[:2])
    with x2_bindings():
        with pytest.raises(ValueError, match="admits only"):
            evaluate_x2(run_dir, 30, family="bc_mlp", expected_sha256=sha, runs_dir=tmp_path,
                        suite=x2s.HELDOUT_SUITE)
        s = evaluate_x2(run_dir, 30, family="bc_mlp", expected_sha256=sha, runs_dir=tmp_path)
        ph_path, ph_sha = _ckpt(runs["plain"])
        ph_dir = runs["plain"]["checkpoints"][-1]["path"].rsplit("/checkpoints/", 1)[0]
        with pytest.raises(CheckpointError):
            evaluate_x2(ph_dir, 30, family="bc_mlp", expected_sha256=ph_sha, runs_dir=tmp_path)
    assert s["suite"] == x2s.VALIDATION_SUITE and s["n_episodes"] == 2
    assert s["episode_rules"]["horizon"] == 500
    assert s["dataset_id"] == "robomimic_v1.5/can/mh" and s["verify_passed"]
    assert s["eligible_for_selection"] is False  # 2 of 50 starts, native: never selectable
    assert set(s) >= {"eligible_for_selection", "n_episodes", "partial", "checkpoint",
                      "eval_run_id", "successes", "first_success_step", "wilson95", "platform",
                      "train_run_id"}  # the fields roboforge.eval.selection reads
