"""X2-a headroom pilot: configuration, run specs and the preregistered reading (D-124).

The pilot trains one seed of each family's frozen chosen Can configuration on the X2 (Can MH)
data and evaluates a stride of checkpoints on X2 validation starts only. Its numbers are
development numbers: one seed, the best of a few checkpoints on the same starts (biased
upward, RESEARCH_PROTOCOL §7), never a result. It answers one question before the X2-b
search space and budget are fixed: is Can MH neither saturated nor at zero for these
configurations (``docs/extensions/X2_harder_task.md`` §3)?

:func:`reading` applies the rule fixed in ``configs/extensions/x2_pilot.yaml`` before the
first pilot launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roboforge.paths import REPO_ROOT, configs_dir
from roboforge.train.config import RunConfig, load_config, run_id

SCHEMA = "roboforge.x2_pilot/1"
CONFIG = "configs/extensions/x2_pilot.yaml"
FAMILIES = ("diffusion_unet", "bc_rnn", "bc_mlp_chunk", "bc_mlp")


def pilot_config() -> dict[str, Any]:
    """Load and validate ``configs/extensions/x2_pilot.yaml``."""
    import yaml

    cfg = yaml.safe_load((configs_dir() / "extensions" / "x2_pilot.yaml").read_text())
    keys = {"schema", "decision", "label", "task", "prefix", "seed", "families", "budget",
            "eval_steps", "suite", "limits", "reading"}
    if set(cfg) != keys or cfg["schema"] != SCHEMA:
        raise ValueError(f"{CONFIG}: keys {sorted(cfg)} or schema")
    if tuple(cfg["families"]) != FAMILIES:
        raise ValueError(f"{CONFIG}: families {list(cfg['families'])} != {FAMILIES}")
    if int(cfg["seed"]) != 0:
        raise ValueError(f"{CONFIG}: the pilot seed must be 0 (KI-062)")
    return cfg


@dataclass(frozen=True)
class PilotRun:
    """One pilot training run."""

    family: str
    config_path: str
    config: RunConfig
    run_id: str


def pilot_runs() -> list[PilotRun]:
    """The four pilot runs, resolved exactly as the frozen suite resolves a trial-1 run
    (budget, then ``task``, then ``seed``; ``roboforge.train.suite.SuitePlan.spec``)."""
    cfg = pilot_config()
    out = []
    for fam, path in cfg["families"].items():
        rc = load_config(REPO_ROOT / path, [*cfg["budget"], f"task={cfg['task']}",
                                            f"seed={int(cfg['seed'])}"])
        if rc.family != ("bc_mlp" if fam == "bc_mlp_chunk" else fam):
            raise ValueError(f"{path}: model family {rc.family!r} for pilot family {fam!r}")
        out.append(PilotRun(fam, path, rc, run_id(rc, cfg["prefix"])))
    return out


def reading(best: dict[str, int], rule: dict[str, Any] | None = None) -> dict[str, Any]:
    """The preregistered headroom reading from each family's best success count (of n)."""
    rule = rule or pilot_config()["reading"]
    if sorted(best) != sorted(FAMILIES):
        raise ValueError(f"need all four families, got {sorted(best)}")
    n = int(rule["n_states"])
    if any(not 0 <= int(v) <= n for v in best.values()):
        raise ValueError(f"success counts must lie in [0, {n}]")
    n_high = sum(int(v) >= int(rule["saturated_threshold"]) for v in best.values())
    if n_high >= int(rule["saturated_min_families"]):
        label = "saturated"
    elif all(int(v) <= int(rule["zero_threshold"]) for v in best.values()):
        label = "at zero"
    else:
        label = "headroom"
    return {"label": label, "best": dict(best), "n_families_at_or_above_saturated": n_high,
            "rule": rule}
