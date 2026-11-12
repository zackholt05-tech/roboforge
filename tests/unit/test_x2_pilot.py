"""The X2-a pilot: frozen configurations, seed 0, the reading rule and the evaluator's suite
admission (DECISIONS D-123, D-124). No dataset or simulator needed."""

from __future__ import annotations

import pytest

from roboforge.eval.freeze import load_freeze
from roboforge.x2 import pilot
from roboforge.x2 import states as x2s
from roboforge.x2.evaluate import check_x2_eval_suite


def test_pilot_runs_are_the_frozen_seed0_can_configurations() -> None:
    frozen = load_freeze(require_committed=False)["tasks"]["can"]["families"]
    runs = pilot.pilot_runs()
    assert [r.family for r in runs] == list(pilot.FAMILIES)
    for r in runs:
        f0 = next(x for x in frozen[r.family]["runs"] if int(x["seed"]) == 0)
        assert r.config.sha256() == f0["config_sha256"]
        assert r.config.task == "can" and r.config.seed == 0
        assert r.run_id == "s15x2_" + f0["train_run_id"].split("_", 1)[1]
        assert r.config.train.max_steps == 100_000 and r.config.train.ckpt_every == 5_000


def test_pilot_config_uses_validation_starts_and_a_stride() -> None:
    cfg = pilot.pilot_config()
    assert cfg["suite"] == x2s.VALIDATION_SUITE
    assert cfg["eval_steps"] == [20_000, 40_000, 60_000, 80_000, 100_000]
    assert cfg["limits"]["max_containers_during_diffusion"] == 2


@pytest.mark.parametrize(("best", "label"), [
    ({"diffusion_unet": 50, "bc_rnn": 49, "bc_mlp_chunk": 48, "bc_mlp": 37}, "saturated"),
    ({"diffusion_unet": 48, "bc_rnn": 47, "bc_mlp_chunk": 48, "bc_mlp": 30}, "headroom"),
    ({"diffusion_unet": 2, "bc_rnn": 0, "bc_mlp_chunk": 1, "bc_mlp": 0}, "at zero"),
    ({"diffusion_unet": 3, "bc_rnn": 0, "bc_mlp_chunk": 1, "bc_mlp": 0}, "headroom"),
    ({"diffusion_unet": 20, "bc_rnn": 35, "bc_mlp_chunk": 30, "bc_mlp": 10}, "headroom"),
])
def test_reading_rule(best, label) -> None:
    assert pilot.reading(best)["label"] == label


def test_reading_needs_every_family_and_valid_counts() -> None:
    with pytest.raises(ValueError, match="four families"):
        pilot.reading({"diffusion_unet": 10})
    with pytest.raises(ValueError, match="must lie"):
        pilot.reading({"diffusion_unet": 51, "bc_rnn": 0, "bc_mlp_chunk": 0, "bc_mlp": 0})


def test_evaluator_admits_only_the_x2_validation_suite() -> None:
    check_x2_eval_suite("can", x2s.VALIDATION_SUITE)
    for suite in (x2s.HELDOUT_SUITE, "validation", "robustness_base"):
        with pytest.raises(ValueError, match="admits only"):
            check_x2_eval_suite("can", suite)
    with pytest.raises(ValueError, match="evaluates can only"):
        check_x2_eval_suite("lift", x2s.VALIDATION_SUITE)
