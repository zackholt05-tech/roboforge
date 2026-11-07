"""Session 7 evaluator extensions (RESEARCH_PROTOCOL §14, D-064): robustness suites and the
preregistered inference variants. Validation IDs and selection eligibility are unchanged.

Needs the simulator and the processed data; the tiny runs trained here are fixtures."""

from __future__ import annotations

import pytest
import torch

from roboforge.eval.learned import (
    VARIANTS,
    LearnedPolicy,
    model_from_payload,
    trained_model_from_payload,
    variant_model,
)
from roboforge.eval.validation import check_suite_allowed, eval_run_id, evaluate_checkpoint
from roboforge.paths import REPO_ROOT
from roboforge.sim import eval_states as es
from roboforge.train.checkpoint import load_checkpoint
from roboforge.train.config import load_config
from roboforge.train.run import train_run

pytestmark = [pytest.mark.sim, pytest.mark.data]
TINY = ["train.max_steps=40", "train.ckpt_every=40", "train.val_every=40", "train.log_every=20"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, lift_lowdim_path):
    root = tmp_path_factory.mktemp("runs")
    out = {}
    for fam in ("diffusion_unet", "bc_mlp"):
        cfg = load_config(REPO_ROOT / "configs" / "models" / f"{fam}.yaml", TINY)
        out[fam] = (train_run(cfg, root, "test"), root)
    return out


def _payload(m):
    c = m["checkpoints"][-1]
    return load_checkpoint(c["path"], expected_sha256=c["sha256"]), c


def test_validation_run_ids_are_unchanged_and_variants_are_suffixed() -> None:
    p = "linux-aarch64-container-1f685638f28c"
    assert eval_run_id("r", 5000, p, stochastic=False) == f"r__val_step0005000__{p}"
    assert eval_run_id("r", 5000, p, stochastic=True) == f"r__val_step0005000__{p}__ps0r0"
    assert (eval_run_id("r", 5000, p, stochastic=True, suite="robustness_base",
                        variant="ddpm100")
            == f"r__robustness_base_step0005000__{p}__ps0r0__var-ddpm100")


def test_only_validation_and_robustness_suites_are_evaluable() -> None:
    check_suite_allowed("lift", "validation")
    for name in ("robustness_base", "ood_cube_shift", "ood_arm_offset"):
        check_suite_allowed("lift", name)
    check_suite_allowed("can", "ood_arm_offset")
    for task, name in (("can", "ood_cube_shift"), ("lift", es.LOCKED_SUITE), ("lift", "x")):
        with pytest.raises(ValueError, match="not evaluable"):
            check_suite_allowed(task, name)


def test_trained_weights_variant_is_the_non_ema_network(runs) -> None:
    payload, _ = _payload(runs["diffusion_unet"][0])
    trained = variant_model(payload, "trained-weights").state_dict()
    ref = trained_model_from_payload(payload).state_dict()
    ema = model_from_payload(payload).state_dict()
    assert all(torch.equal(trained[k], ref[k]) for k in ref)
    assert any(not torch.equal(trained[k], ema[k]) for k in ref)


def test_ddpm100_variant_changes_only_the_sampler(runs) -> None:
    payload, _ = _payload(runs["diffusion_unet"][0])
    primary, ddpm = model_from_payload(payload), variant_model(payload, "ddpm100")
    assert (primary.describe()["sampler"]["sampler"], ddpm.describe()["sampler"]["sampler"]) == (
        "ddim", "ddpm")
    assert len(ddpm.sampler.timesteps) == 100 and len(primary.sampler.timesteps) == 10
    sp, sd = primary.state_dict(), ddpm.state_dict()
    assert sp.keys() == sd.keys() and all(torch.equal(sp[k], sd[k]) for k in sp)


def test_variants_refuse_unknown_names_and_non_diffusion(runs) -> None:
    dpay, _ = _payload(runs["diffusion_unet"][0])
    bpay, _ = _payload(runs["bc_mlp"][0])
    with pytest.raises(ValueError, match="unknown inference variant"):
        variant_model(dpay, "nope")
    with pytest.raises(ValueError, match="diffusion"):
        variant_model(bpay, "trained-weights")
    assert set(VARIANTS) == {"trained-weights", "ddpm100"}


def test_policy_records_the_variant(runs) -> None:
    _, c = _payload(runs["diffusion_unet"][0])
    pol = LearnedPolicy.from_checkpoint(c["path"], expected_sha256=c["sha256"],
                                        variant="trained-weights")
    assert pol.info.params["inference_network"] == "trained"
    assert pol.info.params["inference_variant"]["name"] == "trained-weights"
    assert "ablation" in pol.info.label
    prim = LearnedPolicy.from_checkpoint(c["path"], expected_sha256=c["sha256"])
    assert prim.info.params["inference_network"] == "ema"
    assert "inference_variant" not in prim.info.params


def test_robustness_evaluation_never_selects(runs, tmp_path) -> None:
    if not es.manifest_path("lift", "robustness_base").exists():
        pytest.skip("robustness suites not generated yet (make robustness-states)")
    m, root = runs["diffusion_unet"]
    s = evaluate_checkpoint(root / m["run_id"], m["checkpoints"][-1]["step"],
                            runs_dir=tmp_path, n_states=2, n_verify=1,
                            suite="robustness_base", variant="ddpm100")
    assert s["suite"] == "robustness_base" and s["partition"] == "eval_robustness"
    assert s["inference_variant"] == "ddpm100" and not s["eligible_for_selection"]
    assert s["schema"] == "roboforge.robustness_eval/1" and s["verify_passed"]
