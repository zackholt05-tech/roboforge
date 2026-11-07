"""Modal list prices and resource-time cost arithmetic (Session 5).

Every value here is an **estimate** from the posted list prices in
``configs/compute/modal.yaml`` (copied from https://modal.com/pricing with its fetch date). A
bill can differ: credits, region and non-preemptible multipliers, billed keep-alive, and
charges this model does not include. Billed amounts come only from Modal's own records
(:mod:`roboforge.compute.cost_log`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from roboforge.paths import configs_dir

MODAL_CONFIG = configs_dir() / "compute" / "modal.yaml"


def load_modal_config(path: Path = MODAL_CONFIG) -> dict[str, Any]:
    """The committed Modal settings and price snapshot."""
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema") != "roboforge.modal_config/1":
        raise ValueError(f"{path}: not a roboforge.modal_config/1 file")
    return cfg


@dataclass(frozen=True)
class Resources:
    """What one container holds for ``seconds`` (cores are physical cores)."""

    seconds: float
    cpu: float
    memory_gib: float
    gpu: str | None = None
    count: int = 1  # identical containers

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {"seconds": self.seconds, "cpu": self.cpu, "memory_gib": self.memory_gib,
                "gpu": self.gpu, "count": self.count}


def cost_usd(r: Resources, pricing: dict[str, Any], multiplier: float = 1.0) -> dict[str, Any]:
    """List-price cost of ``r``, split by resource (USD, estimate)."""
    if r.cpu < 0.125:
        raise ValueError("Modal bills at least 0.125 physical cores per container")
    gpu = 0.0
    if r.gpu is not None:
        rates = pricing["gpu_usd_per_s"]
        if r.gpu not in rates:
            raise KeyError(f"no posted price for GPU {r.gpu!r}; known: {sorted(rates)}")
        gpu = rates[r.gpu] * r.seconds
    cpu = pricing["cpu_usd_per_core_s"] * r.cpu * r.seconds
    mem = pricing["memory_usd_per_gib_s"] * r.memory_gib * r.seconds
    k = r.count * multiplier
    return {"gpu_usd": gpu * k, "cpu_usd": cpu * k, "memory_usd": mem * k,
            "total_usd": (gpu + cpu + mem) * k, "multiplier": multiplier,
            "resources": r.to_dict(), "kind": "estimate (list price)"}


def function_resources(cfg: dict[str, Any], function: str, seconds: float) -> Resources:
    """Resources of a configured Modal function held for ``seconds``."""
    f = cfg["functions"][function]
    return Resources(seconds=float(seconds), cpu=float(f["cpu"]),
                     memory_gib=float(f["memory_mib"]) / 1024.0, gpu=f["gpu"])


def launch_upper_bound(cfg: dict[str, Any], function: str, timeout_s: float) -> dict[str, Any]:
    """Worst case of one launch: every allowed attempt held for its full timeout plus the
    billed keep-alive, plus the build/start allowance.

    Modal reschedules an interrupted input even with ``retries=0`` and gives each attempt its
    own timeout (KNOWN_ISSUES KI-031). The job caps attempts per launch
    (``policy.max_attempts_per_launch``), so the bound covers that many attempts."""
    pricing, pol = cfg["pricing"], cfg["policy"]
    attempts = int(pol["max_attempts_per_launch"])
    one = cost_usd(function_resources(cfg, function, timeout_s + pol["scaledown_window_s"]),
                   pricing)
    job = {k: one[k] * attempts for k in ("gpu_usd", "cpu_usd", "memory_usd", "total_usd")}
    job.update(attempts=attempts, resources_per_attempt=one["resources"], kind=one["kind"])
    b = cfg["build_allowance"]
    build = cost_usd(Resources(float(b["seconds"]), float(b["cpu"]), float(b["memory_gib"])),
                     pricing)
    return {"function": function, "timeout_s": timeout_s, "attempts": attempts,
            "one_attempt": one, "job": job, "build_allowance": build,
            "total_usd": job["total_usd"] + build["total_usd"],
            "kind": f"upper-bound estimate (list price; {attempts} attempts of the full timeout)",
            "pricing_fetched_utc": pricing["fetched_utc"], "pricing_source": pricing["source"]}


def storage_month_usd(gib: float, pricing: dict[str, Any]) -> float:
    """Volume storage cost for one month beyond the free allowance (estimate)."""
    return max(0.0, gib - pricing["volume_free_gib"]) * pricing["volume_usd_per_gib_month"]


__all__ = ["MODAL_CONFIG", "Resources", "cost_usd", "function_resources", "launch_upper_bound",
           "load_modal_config", "storage_month_usd"]
