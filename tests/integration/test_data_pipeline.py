"""The canonical data pipeline on the real robomimic PH data (skipped when not downloaded).

Covers spec/IDs vs the dataset and the pinned simulator, recomputed features vs stored
observations (KI-003), committed-artifact integrity and determinism, train-only
normalization, real-data windows with the future-leakage check, and a one-batch forward
pass for Lift and Can.
"""

from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from roboforge.data.features import FeatureExtractor
from roboforge.data.normalization import Normalizer
from roboforge.data.pipeline import (
    build_task,
    check_task,
    load_specs,
    load_task_artifacts,
    write_task,
)
from roboforge.data.processed import (
    EQUAL_ATOL,
    ROBOSUITE_F32_ATOL,
    build_episodes,
    load_processed,
    processed_path,
)
from roboforge.data.sequence import SequenceDataset, collate
from roboforge.data.specs import build_action_spec, build_observation_spec, load_data_config
from roboforge.data.split import upstream_split
from roboforge.provenance import canonical_json
from roboforge.sim.factory import load_sim_config, make_env
from roboforge.sim.state_replay import reset_to_state
from tests.unit.test_forward_pass import one_batch_forward

pytestmark = pytest.mark.data
TASKS = ["lift", "can"]


@pytest.fixture(scope="module", params=TASKS)
def task(request) -> str:
    """Task name, skipping when its dataset or processed file is absent."""
    request.getfixturevalue(f"{request.param}_lowdim_path")
    m = load_task_artifacts(request.param)["task_manifest"]
    if not processed_path(request.param, m["processed"]["converter_version"]).exists():
        pytest.skip("processed data not built (run `make build-data`)")
    return request.param


@pytest.fixture(scope="module")
def low(task, request):
    """Open low_dim HDF5 file for the task."""
    with h5py.File(request.getfixturevalue(f"{task}_lowdim_path"), "r") as f:
        yield f


@pytest.fixture(scope="module")
def env(task):
    """Pinned simulator built from the task's dataset metadata."""
    e = make_env(load_sim_config(task))
    yield e
    e.close()


def test_specs_from_hdf5_equal_committed_specs(task, low) -> None:
    obs, act = load_specs(task)
    assert build_observation_spec(load_data_config(task), low).sha256() == obs.sha256()
    assert build_action_spec(low) == act
    assert obs.dim == 19 and act.dim == 7 and obs.privileged


def test_action_spec_matches_simulator(task, env) -> None:
    _, act = load_specs(task)
    lo, hi = env.action_spec
    assert lo.tolist() == list(act.low) and hi.tolist() == list(act.high)
    assert env.action_dim == act.dim


def test_feature_ids_equal_robosuite_ids(task, env, low) -> None:
    g = low["data"]["demo_0"]
    reset_to_state(env, g.attrs["model_file"], g["states"][0])
    obs_spec, _ = load_specs(task)
    ids = FeatureExtractor(obs_spec).bind(env.sim.model._model).resolved_ids()
    robot = env.robots[0]
    assert ids["robot0_eef_pos"]["site"] == robot.eef_site_id["right"]
    assert ids["robot0_eef_quat"]["body"] == env.sim.model.body_name2id(
        robot.robot_model.eef_name["right"])
    assert ids["robot0_gripper_qpos"]["qpos_adr"] == list(
        robot._ref_gripper_joint_pos_indexes["right"])
    obj = env.cube_body_id if task == "lift" else env.obj_body_id["Can"]
    assert ids[obs_spec.features[-1].name]["body"] == obj


@pytest.mark.parametrize("which", ["shortest", "longest", "first", "valid"])
def test_recomputed_features_equal_dataset(task, env, low, which) -> None:
    m = load_task_artifacts(task)["task_manifest"]
    lengths = {e["id"]: e["length"] for e in m["episodes"]}
    valid = [e["id"] for e in m["episodes"] if e["split"] == "valid"]
    demo = {"shortest": min(lengths, key=lengths.get), "longest": max(lengths, key=lengths.get),
            "first": "demo_0", "valid": valid[-1]}[which]
    obs_spec, _ = load_specs(task)
    episodes, ver = build_episodes(env, low, obs_spec, demos=[demo])
    assert all(v <= EQUAL_ATOL for v in ver.max_abs_vs_dataset.values())
    assert ver.final_state_success == {demo: True}
    if task == "can":  # KI-003: corrected, and the stored block really is one update stale
        c = ver.corrected["Can_to_robot0_eef_pos"]
        assert c["max_abs_vs_float64_same_step_reference"] <= EQUAL_ATOL
        assert c["max_abs_vs_robosuite_float32_formula"] <= ROBOSUITE_F32_ATOL
        # robosuite's float32 rotation is platform-dependent under numba (KI-018)
        assert c["stored_vs_robosuite_lag_formula_max_t_gt0"] <= ROBOSUITE_F32_ATOL
        assert c["stored_minus_corrected_max_t0"] > 1e-2
    processed = load_processed(processed_path(task, "lowdim_v1"), demos=[demo])
    np.testing.assert_array_equal(processed[demo]["obs"], episodes[demo]["obs"])


def test_corrected_can_feature_equals_robosuite_with_fresh_cache(can_lowdim_path) -> None:
    obs_spec, _ = load_specs("can")
    ext = FeatureExtractor(obs_spec)
    env = make_env(load_sim_config("can"))
    with h5py.File(can_lowdim_path, "r") as f:
        g = f["data"]["demo_3"]
        states = g["states"][()]
        reset_to_state(env, g.attrs["model_file"], states[0])
    for t in (0, 1, 40, len(states) - 1):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        env._get_observations(force_update=True)
        fresh = env._get_observations(force_update=True)
        feats = ext.dict_from_sim(env.sim)
        np.testing.assert_allclose(feats["Can_to_robot0_eef_pos"], fresh["object-state"][0:3],
                                   rtol=0, atol=ROBOSUITE_F32_ATOL)  # robosuite: float32 rotation
        np.testing.assert_array_equal(feats["Can_pos"], fresh["Can_pos"])
    env.close()


def test_committed_artifacts_and_processed_hash(task) -> None:
    docs = load_task_artifacts(task)
    m = docs["task_manifest"]
    eps = load_processed(processed_path(task, "lowdim_v1"),
                         expected_sha256=m["processed"]["content_sha256"])
    assert len(eps) == 200 == m["task_spec"]["n_demos"]
    assert sum(len(e["actions"]) for e in eps.values()) == m["task_spec"]["total_samples"]
    assert all(e["obs"].dtype == np.float32 and e["obs"].shape[1] == 19 for e in eps.values())


def test_split_is_the_upstream_mask_partition_and_reproducible(task, low) -> None:
    sp = load_task_artifacts(task)["split"]
    train, valid = sp["splits"]["train"], sp["splits"]["valid"]
    assert len(train) == 180 and len(valid) == 20 and not set(train) & set(valid)
    assert set(train) | set(valid) == set(low["data"].keys())
    assert set(valid) == {x.decode() for x in low["mask/valid"][()]}
    assert train == sorted(train, key=lambda d: int(d.split("_")[1]))
    rebuilt = upstream_split(low, task, sp["source_low_dim_sha256"], flagged=sp["flagged"])
    assert canonical_json(rebuilt) == canonical_json(sp)


def test_normalization_is_fit_on_train_episodes_only(task) -> None:
    docs = load_task_artifacts(task)
    sp, norm = docs["split"], docs["normalization"]
    eps = load_processed(processed_path(task, "lowdim_v1"))
    tr = sp["splits"]["train"]
    refit = Normalizer.fit([eps[d]["obs"] for d in tr], "minmax").to_dict()
    assert refit == norm["obs"]
    tampered = copy.deepcopy(eps)
    for d in sp["splits"]["valid"]:
        tampered[d]["obs"] += 123.0
        tampered[d]["actions"] *= -1
    assert Normalizer.fit([tampered[d]["obs"] for d in tr], "minmax").to_dict() == norm["obs"]
    assert Normalizer.fit([tampered[d]["actions"] for d in tr], "minmax").to_dict() == \
        norm["action"]
    with_valid = Normalizer.fit([eps[d]["obs"] for d in eps], "minmax").to_dict()
    assert with_valid != norm["obs"]


def test_normalizer_inverts_real_data(task) -> None:
    docs = load_task_artifacts(task)
    eps = load_processed(processed_path(task, "lowdim_v1"))
    for key, field in (("obs", "obs"), ("action", "actions")):
        n = Normalizer.from_dict(docs["normalization"][key])
        x = np.concatenate([e[field] for e in eps.values()]).astype(np.float64)
        np.testing.assert_allclose(n.unnormalize(n.normalize(x)), x, rtol=0, atol=1e-12)


def test_windows_on_real_data_stay_in_split_and_do_not_leak(task) -> None:
    train = SequenceDataset.from_committed(task, "train")
    valid = SequenceDataset.from_committed(task, "valid")
    sp = load_task_artifacts(task)["split"]
    assert len(train) == sp["samples"]["train"] and len(valid) == sp["samples"]["valid"]
    assert {d for d, _ in train.index} == set(sp["splits"]["train"])
    assert not {d for d, _ in train.index} & {d for d, _ in valid.index}
    demo = min(train.demos, key=lambda d: len(train._act[d]))
    starts = [i for i, (d, _) in enumerate(train.index) if d == demo]
    n = len(starts)
    first, mid, last = train[starts[0]], train[starts[n // 2]], train[starts[-1]]
    assert first["obs_valid"].tolist() == [False, True] and first["valid_mask"].all()
    assert mid["obs_valid"].all() and mid["obs_index"].tolist() == [n // 2 - 1, n // 2]
    assert last["valid_mask"].sum() == 1 and last["action_index"].tolist() == [n - 1] * 16
    np.testing.assert_array_equal(mid["obs_history"], train._obs[demo][[n // 2 - 1, n // 2]])
    before = mid["obs_history"].copy()
    train._obs[demo][n // 2 + 1:] += 50.0  # corrupt every future observation
    np.testing.assert_array_equal(train[starts[n // 2]]["obs_history"], before)


def test_one_batch_forward_on_real_data(task) -> None:
    ds = SequenceDataset.from_committed(task, "train")
    picks = np.linspace(0, len(ds) - 1, 8).astype(int).tolist()
    sub = [ds[i] for i in picks]
    b = collate(sub)
    assert tuple(b["obs_history"].shape) == (8, 2, 19)
    assert tuple(b["future_action"].shape) == (8, 16, 7)
    assert np.isfinite(b["obs_history"].numpy()).all()
    assert np.abs(b["future_action"].numpy()).max() <= 1.0 + 1e-6
    ds.index = [ds.index[i] for i in picks]
    pred, loss = one_batch_forward(ds)
    assert pred.shape == (8, 16, 7) and np.isfinite(pred.numpy()).all() and np.isfinite(loss)


def test_can_rebuild_reproduces_committed_artifacts(can_lowdim_path) -> None:
    a, b = build_task("can"), build_task("can")
    assert a.task_manifest == b.task_manifest and a.normalization == b.normalization
    assert check_task(a) == []


def test_build_refuses_to_overwrite_differing_committed_artifacts(can_lowdim_path,
                                                                tmp_path) -> None:
    art = build_task("can")
    art.split = {**art.split, "notes": "tampered"}
    with pytest.raises(ValueError, match="refusing to write"):
        write_task(art, processed_root=tmp_path)
    assert not list(tmp_path.rglob("*.hdf5"))


@pytest.mark.slow
def test_lift_rebuild_reproduces_committed_artifacts(lift_lowdim_path) -> None:
    assert check_task(build_task("lift")) == []
