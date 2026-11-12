"""X2 (Can MH) data artifacts: integrity, operator-stratified split, accepted flags (D-119-D-122).

The committed-file tests need no dataset. The rebuild test needs the downloaded MH files and
the simulator, and skips without them.
"""

from __future__ import annotations

import json

import pytest

from roboforge.data import pipeline
from roboforge.x2 import data as x2


def test_committed_x2_artifacts_load_and_keep_the_can_ph_specs() -> None:
    docs = x2.load_x2_artifacts()
    m = docs["task_manifest"]
    ph = pipeline.load_task_artifacts("can")["task_manifest"]
    assert m["dataset_id"] == "robomimic_v1.5/can/mh" == docs["normalization"]["dataset_id"]
    for k in ("observation_spec_sha256", "action_spec_sha256"):
        assert m[k] == ph[k]
    assert m["observation_spec"] == ph["observation_spec"]
    assert m["action_spec"] == ph["action_spec"]
    assert m["task_spec"]["eval_horizon_steps"] == x2.EVAL_HORIZON_STEPS == 500
    assert m["task_spec"]["state_dim"] == ph["task_spec"]["state_dim"] == 71
    assert m["source"]["files"]["low_dim"]["sha256"].startswith("c4a34c83")
    assert docs["normalization"]["split_content_sha256"] == docs["split"]["content_sha256"]


def test_split_is_whole_episode_and_operator_stratified() -> None:
    docs = x2.load_x2_artifacts()
    eps = docs["task_manifest"]["episodes"]
    sp = docs["split"]
    assert len(eps) == 300 and sp["counts"] == {"train": 270, "valid": 30, "total": 300}
    assert not set(sp["splits"]["train"]) & set(sp["splits"]["valid"])
    ops = {e["operator"] for e in eps}
    assert len(ops) == 6
    for op in ops:
        mine = [e for e in eps if e["operator"] == op]
        assert len(mine) == 50
        assert sum(e["split"] == "train" for e in mine) == 45
        assert {e["quality"] for e in mine} == {op.split("_operator_")[0]}
    for e in eps:
        assert e["id"] in sp["splits"][e["split"]]


def test_accepted_flags_need_the_exact_reason() -> None:
    reason = x2.ACCEPTED_FLAGS["demo_81"]["reason"]
    assert x2.unaccepted_flags({"demo_81": reason}) == []
    assert x2.unaccepted_flags({"demo_81": "something else"}) == ["demo_81"]
    assert x2.unaccepted_flags({"demo_7": reason}) == ["demo_7"]
    sp = x2.load_x2_artifacts()["split"]
    assert sp["flagged"] == {"demo_81": reason}
    assert sp["accepted_flags"] == {"demo_81": x2.ACCEPTED_FLAGS["demo_81"]}


def test_x2_paths_are_new_and_the_loader_refuses_other_tasks() -> None:
    mine = set(x2.committed_paths().values())
    for task in ("lift", "can"):
        assert not mine & set(pipeline.committed_paths(task, x2.CONVERTER).values())
    assert "_ph" not in x2.processed_relpath()
    with pytest.raises(ValueError, match="X2 data exist for can"):
        x2.load_x2_artifacts("lift")


def test_loader_refuses_an_edited_manifest(tmp_path, monkeypatch) -> None:
    paths = x2.committed_paths()
    edited = {}
    for k, p in paths.items():
        q = tmp_path / k / p.name
        q.parent.mkdir()
        q.write_text(p.read_text())
        edited[k] = q
    doc = json.loads(edited["task_manifest"].read_text())
    doc["task_spec"]["eval_horizon_steps"] = 400
    edited["task_manifest"].write_text(json.dumps(doc))
    monkeypatch.setattr(x2, "committed_paths", lambda: edited)
    with pytest.raises(ValueError, match="content hash mismatch"):
        x2.load_x2_artifacts()


@pytest.mark.sim
@pytest.mark.data
def test_rebuild_equals_the_committed_artifacts() -> None:
    _, files = __import__("roboforge.data.fetch", fromlist=["load_manifest"]).load_manifest(
        x2.X2_MANIFEST)
    if not all(f.destination().exists() for f in files):
        pytest.skip("Can MH files not downloaded (run `make x2-download`)")
    from roboforge.sim.backend import configure_mujoco_gl

    configure_mujoco_gl()
    art = x2.build_x2()
    assert x2.check_x2(art) == []
    assert x2.unaccepted_flags(art.split["flagged"]) == []
