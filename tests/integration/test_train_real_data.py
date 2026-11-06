"""Training on the real robomimic PH data: deliberate overfit, run manifest, reload.

Skipped when the processed data are not built (``make build-data``). The overfit checks are
unit/integration sanity tests of the learning pipeline (MASTER_SPEC §5.1), not results.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.processed import processed_path
from roboforge.data.sequence import SequenceDataset
from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file
from roboforge.train.config import load_config
from roboforge.train.data import WindowTensors
from roboforge.train.loop import Trainer, offline_metrics
from roboforge.train.run import data_context, load_windows, train_run, write_manifest

pytestmark = pytest.mark.data
CONFIGS = REPO_ROOT / "configs" / "models"


@pytest.fixture(scope="module", autouse=True)
def _processed(lift_lowdim_path) -> None:
    m = load_task_artifacts("lift")["task_manifest"]
    if not processed_path("lift", m["processed"]["converter_version"]).exists():
        pytest.skip("processed data not built (run `make build-data`)")


def test_windows_are_the_committed_demonstration_data() -> None:
    cfg = load_config(CONFIGS / "bc_rnn.yaml")
    train, valid = load_windows(cfg)
    ds = SequenceDataset.from_committed("lift", "train", obs_horizon=10, pred_horizon=1)
    for i in (0, 1234, len(ds) - 1):
        item = ds[i]
        np.testing.assert_array_equal(train.obs_history[i].numpy(), item["obs_history"])
        np.testing.assert_array_equal(train.obs_valid[i].numpy(), item["obs_valid"])
        np.testing.assert_array_equal(train.future_action[i].numpy(), item["future_action"])
    split = load_task_artifacts("lift")["split"]["splits"]
    assert set(train.episode_id) == set(split["train"]) and len(set(train.episode_id)) == 180
    assert set(valid.episode_id) == set(split["valid"]) and len(set(valid.episode_id)) == 20
    assert len(train) + len(valid) == 9666  # every Lift sample, one window per step


@pytest.mark.parametrize(("config", "steps", "ratio"), [("bc_mlp", 600, 0.1),
                                                        ("bc_rnn", 300, 0.2)])
def test_tiny_episode_overfit_on_real_demos(config: str, steps: int, ratio: float) -> None:
    """Deliberate overfit of 2 training demos with the primary architectures."""
    cfg = load_config(CONFIGS / f"{config}.yaml",
                      [f"train.max_steps={steps}", f"train.ckpt_every={steps}",
                       "train.batch_size=50", "optim.lr=1.0e-3"])
    train, _ = load_windows(cfg)
    tiny = train.subset({"demo_1", "demo_2"})
    tr = Trainer(cfg, tiny, None, data_context("lift", "test"))
    before = offline_metrics(tr.model, tiny)["mse"]
    for _ in range(steps):
        tr.step()
    after = offline_metrics(tr.model, tiny)["mse"]
    assert after < ratio * before, (before, after)


def test_tiny_batch_overfit_on_real_windows() -> None:
    cfg = load_config(CONFIGS / "bc_mlp.yaml", ["train.batch_size=32", "optim.lr=1.0e-3",
                                               "train.max_steps=300", "train.ckpt_every=300"])
    train, _ = load_windows(cfg)
    idx = np.arange(0, len(train), len(train) // 32)[:32]
    batch = WindowTensors(train.obs_history[idx], train.obs_valid[idx],
                          train.future_action[idx], train.valid_mask[idx],
                          tuple(train.episode_id[i] for i in idx), train.start_t[idx])
    tr = Trainer(cfg, batch, None, data_context("lift", "test"))
    first = tr.step()["loss"]
    for _ in range(299):
        last = tr.step()["loss"]
    assert last < 0.02 * first, (first, last)


def test_train_run_writes_an_immutable_complete_manifest(tmp_path) -> None:
    cfg = load_config(CONFIGS / "bc_mlp_chunk.yaml",
                      ["train.max_steps=40", "train.ckpt_every=20", "train.val_every=20",
                       "train.log_every=10"])
    m = train_run(cfg, tmp_path, "test")
    run_dir = tmp_path / m["run_id"]
    assert m["status"] == "complete" and m["reload_check"]["passed"]
    assert [c["step"] for c in m["checkpoints"]] == [20, 40]
    for c in m["checkpoints"]:
        assert sha256_file(c["path"]) == c["sha256"]
    committed = load_task_artifacts("lift")
    assert m["data"]["normalization_sha256"] == committed["normalization"]["content_sha256"]
    assert m["data"]["split_sha256"] == committed["split"]["content_sha256"]
    assert m["param_count"] == 1_204_336 and m["dims"]["exec_horizon"] == 8
    assert m["git"]["sha"] and m["environment"]["packages"]["torch"]
    on_disk = json.loads((run_dir / "manifest.json").read_text())
    assert on_disk["run_id"] == m["run_id"] and on_disk["status"] == "complete"
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"train", "offline"}
    assert all(np.isfinite(r["grad_norm_max"]) for r in rows if r["kind"] == "train")
    with pytest.raises(FileExistsError):
        write_manifest(run_dir / "manifest.json", m)  # manifests are written once
    with pytest.raises(FileExistsError):
        train_run(cfg, tmp_path, "test")  # run directories are never reused
    payload = torch.load(m["checkpoints"][-1]["path"], weights_only=True)
    assert payload["hashes"] == m["data"] and payload["global_step"] == 40
