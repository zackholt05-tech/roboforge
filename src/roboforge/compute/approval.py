"""Owner approvals for paid compute, read from ``docs/EXPERIMENT_LEDGER.md`` (Session 5).

A paid Modal launch needs an approval row that the owner granted and that is recorded in the
ledger's table "Owner approvals for paid compute":

    | A-01 | 2026-11-06 | Gate A | gpu_probe, train_cpu | 2.00 | granted | <owner's words> |

The columns are the ID, date, gate, the functions it covers, the cap in USD, the status
(``granted``, ``revoked`` or ``exhausted``) and the evidence. The gate refuses a launch unless:

* the approval exists, is ``granted`` and covers the function;
* the spend already committed under it (:func:`roboforge.compute.cost_log.committed_usd`),
  plus this launch's upper-bound estimate, stays within the cap.

The gate is a client-side guard, not a financial hard stop. A preemption restart or an
unmodelled charge can exceed an estimate. Only a Modal workspace budget caps spending.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboforge.compute.cost_log import committed_usd
from roboforge.paths import REPO_ROOT

LEDGER_DOC = REPO_ROOT / "docs" / "EXPERIMENT_LEDGER.md"
ROW = re.compile(r"^\|\s*(A-\d{2,})\s*\|\s*([0-9-]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
                 r"\s*\$?([0-9.]+)\s*\|\s*(granted|revoked|exhausted|requested)\s*\|"
                 r"\s*(.*?)\s*\|\s*$")


class ApprovalError(PermissionError):
    """A paid launch that no recorded owner approval covers."""


@dataclass(frozen=True)
class Approval:
    """One recorded owner approval."""

    approval_id: str
    date: str
    gate: str
    functions: tuple[str, ...]
    cap_usd: float
    status: str
    evidence: str


def read_approvals(path: Path = LEDGER_DOC) -> dict[str, Approval]:
    """Every approval row of the ledger, by ID (a duplicate ID raises)."""
    out: dict[str, Approval] = {}
    for line in Path(path).read_text().splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        aid = m.group(1)
        if aid in out:
            raise ValueError(f"duplicate approval {aid} in {path}")
        out[aid] = Approval(aid, m.group(2), m.group(3),
                            tuple(f.strip() for f in m.group(4).split(",") if f.strip()),
                            float(m.group(5)), m.group(6), m.group(7))
    return out


def authorize(approval_id: str | None, function: str, estimate_upper_usd: float,
              events: list[dict[str, Any]], path: Path = LEDGER_DOC) -> dict[str, Any]:
    """Check a paid launch against its approval; return the decision or raise."""
    if not approval_id:
        raise ApprovalError("a paid launch needs --budget-approved <approval ID> from "
                            "docs/EXPERIMENT_LEDGER.md")
    approvals = read_approvals(path)
    a = approvals.get(approval_id)
    if a is None:
        raise ApprovalError(f"approval {approval_id} is not recorded in {path.name}")
    if a.status != "granted":
        raise ApprovalError(f"approval {approval_id} is {a.status}")
    if function not in a.functions:
        raise ApprovalError(f"approval {approval_id} covers {a.functions}, not {function}")
    spent = committed_usd(events, approval_id=approval_id)
    if spent + estimate_upper_usd > a.cap_usd + 1e-12:
        raise ApprovalError(f"approval {approval_id}: committed ${spent:.4f} + this launch's "
                            f"upper bound ${estimate_upper_usd:.4f} exceeds the cap "
                            f"${a.cap_usd:.2f}")
    return {"approval_id": a.approval_id, "gate": a.gate, "cap_usd": a.cap_usd,
            "committed_before_usd": spent, "estimate_upper_usd": estimate_upper_usd,
            "remaining_after_usd": a.cap_usd - spent - estimate_upper_usd}


__all__ = ["Approval", "ApprovalError", "LEDGER_DOC", "authorize", "read_approvals"]
