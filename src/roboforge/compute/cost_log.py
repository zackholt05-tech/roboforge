"""Machine-readable cost and time log of every remote-compute action (Session 5).

``reports/compute/cost_ledger.jsonl`` is append-only and committed. It is cumulative across
sessions, so the spend under an approval and in a calendar month can always be summed. One
JSON object per line, ``schema: roboforge.cost_event/1``, with ``event`` one of:

* ``dry_run``: a plan was printed; nothing was created on Modal (cost 0);
* ``launch``: an approved launch started. It records the approval ID, launch and run IDs,
  function, resources, timeout and the **upper-bound estimate**;
* ``result``: the launch ended. It records the status, the container time measured inside
  the job, and the list-price cost of that measured time (an estimate);
* ``billing``: a billed amount from Modal's own records (``modal billing report`` or the
  dashboard, named in ``source``), entered by reconciliation and never computed.

Amounts are USD. ``kind`` says whether a number is an estimate or billed.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT

# ROBOFORGE_COST_LEDGER redirects the ledger (tests and dry-run rehearsals only).
LEDGER = Path(os.environ.get("ROBOFORGE_COST_LEDGER",
                             REPO_ROOT / "reports" / "compute" / "cost_ledger.jsonl"))
SCHEMA = "roboforge.cost_event/1"
EVENTS = ("dry_run", "launch", "result", "billing")


def append_event(event: str, payload: dict[str, Any], path: Path = LEDGER) -> dict[str, Any]:
    """Append one event (fsynced) and return it."""
    if event not in EVENTS:
        raise ValueError(f"unknown cost event {event!r}")
    row = {"schema": SCHEMA, "event": event, "utc": datetime.now(tz=UTC).isoformat(),
           **payload}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return row


def read_events(path: Path = LEDGER) -> list[dict[str, Any]]:
    """Every event, oldest first (empty if the ledger does not exist yet)."""
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def launch_cost(launch_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Best known cost of one launch: billed if reconciled, else the measured-time estimate,
    else the launch's upper bound (a launch with no result may still be running)."""
    billed = [e for e in events if e["event"] == "billing" and e.get("launch_id") == launch_id]
    if billed:
        return {"usd": sum(e["billed_usd"] for e in billed), "basis": "billed"}
    res = [e for e in events if e["event"] == "result" and e["launch_id"] == launch_id]
    if res and res[-1].get("measured_cost_usd") is not None:
        return {"usd": res[-1]["measured_cost_usd"], "basis": "measured time x list price"}
    la = [e for e in events if e["event"] == "launch" and e["launch_id"] == launch_id]
    if not la:
        raise KeyError(f"no launch {launch_id}")
    return {"usd": la[-1]["estimate_upper_usd"], "basis": "upper-bound estimate"}


def committed_usd(events: list[dict[str, Any]], *, approval_id: str | None = None,
                  month: str | None = None) -> float:
    """Spend counted against a budget: every launch (optionally under one approval, or in a
    ``YYYY-MM`` UTC month), each at its best known cost."""
    total = 0.0
    for e in events:
        if e["event"] != "launch":
            continue
        if approval_id is not None and e.get("approval_id") != approval_id:
            continue
        if month is not None and not e["utc"].startswith(month):
            continue
        total += launch_cost(e["launch_id"], events)["usd"]
    return total


def launched_run_ids(events: list[dict[str, Any]]) -> set[str]:
    """Run IDs that any launch has used (a new launch must not reuse one)."""
    return {e["run_id"] for e in events if e["event"] == "launch" and e.get("run_id")}


__all__ = ["EVENTS", "LEDGER", "append_event", "committed_usd", "launch_cost",
           "launched_run_ids", "read_events"]
