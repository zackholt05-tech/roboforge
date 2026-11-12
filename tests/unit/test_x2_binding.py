"""The X2 data binding rebinds exactly the declared names, only inside the context (D-121)."""

from __future__ import annotations

import importlib

import pytest

from roboforge.x2 import binding
from roboforge.x2.data import load_x2_artifacts


def _current() -> dict[tuple[str, str], object]:
    sites = (*binding.LOADER_SITES, binding.WINDOWS_SITE)
    return {(m, a): getattr(importlib.import_module(m), a) for m, a in sites}


def test_names_are_rebound_inside_and_restored_after() -> None:
    before = _current()
    assert not binding.active()
    with binding.x2_bindings():
        assert binding.active()
        inside = _current()
        for site in binding.LOADER_SITES:
            assert inside[site] is load_x2_artifacts
        assert inside[binding.WINDOWS_SITE] is not before[binding.WINDOWS_SITE]
    assert _current() == before
    assert not binding.active()


def test_names_are_restored_after_an_error_and_the_context_is_not_reentrant() -> None:
    before = _current()
    with pytest.raises(RuntimeError, match="boom"), binding.x2_bindings():
        raise RuntimeError("boom")
    assert _current() == before
    with binding.x2_bindings(), pytest.raises(RuntimeError, match="re-entrant"), \
            binding.x2_bindings():
        pass
    assert _current() == before


def test_frozen_consumers_see_the_x2_artifacts_inside() -> None:
    from roboforge.data import pipeline

    ph = pipeline.load_task_artifacts("can")["task_manifest"]
    with binding.x2_bindings():
        m = pipeline.load_task_artifacts("can")["task_manifest"]
        obs, act = pipeline.load_specs("can")
    assert m["dataset_id"] == "robomimic_v1.5/can/mh" != ph["dataset_id"]
    assert obs.sha256() == ph["observation_spec_sha256"]
    assert act.sha256() == ph["action_spec_sha256"]
    assert pipeline.load_task_artifacts("can")["task_manifest"]["dataset_id"] == ph["dataset_id"]
