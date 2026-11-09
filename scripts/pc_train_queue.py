"""Train the 3 Level 2 point-cloud runs on Modal, strictly one launch after another (Session 9).

The RGB study's queue (``scripts/rgb_train_queue.py``, D-078) for the one Level 2 arm. Run in
the project venv; each launch goes through ``infra/modal_app.py train-gpu-pc`` under the Modal
SDK's Python (D-045):

    uv run --frozen python scripts/pc_train_queue.py --budget-approved A-04 \
        [--modal-python /opt/anaconda3/bin/python] [--only 0]

The order is seeds 0, 1, 2.

How launches proceed:
* A run's first launch is ``mode: new``. Every later one is ``resume``, chosen from the cost
  ledger's launched run IDs, so restarting the queue continues where it stopped.
* Every launch is authorized by the approval gate against the spend committed so far, and
  its result is recorded (``record-result``) before the next starts.
* The queue stops at the first launch that does not end ``complete`` or ``interrupted``, and
  after 4 launches of one run.

Nothing here chooses or changes a configuration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from roboforge.compute import cost_log
from roboforge.compute.pc_train import train_plan
from roboforge.paths import REPO_ROOT

ORDER = [0, 1, 2]
MAX_LAUNCHES_PER_RUN = 4


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budget-approved", required=True)
    p.add_argument("--prefix", default="s9")
    p.add_argument("--modal-python", default="/opt/anaconda3/bin/python")
    p.add_argument("--only", nargs="*", type=int, help="seeds, e.g. 0")
    p.add_argument("--log", type=Path, default=REPO_ROOT / "logs" / "session9" / "pc_train" /
                   "queue.jsonl")
    a = p.parse_args(argv)
    order = ORDER if not a.only else list(a.only)
    a.log.parent.mkdir(parents=True, exist_ok=True)
    for seed in order:
        rid = train_plan(seed, a.prefix, None, events=[])["run_id"]
        for n in range(MAX_LAUNCHES_PER_RUN):
            launched = rid in cost_log.launched_run_ids(cost_log.read_events())
            mode = "resume" if launched else "new"
            cmd = [a.modal_python, str(REPO_ROOT / "infra" / "modal_app.py"), "train-gpu-pc",
                   "--seed", str(seed), "--prefix", a.prefix,
                   "--mode", mode, "--budget-approved", a.budget_approved]
            t0 = time.time()
            launch_log = a.log.parent / f"{rid}_launch{n}.log"
            with launch_log.open("w") as fh:  # streamed, so a hung launch is visible
                proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                      text=True)
            out = launch_log.read_text()
            m = re.search(r"^result: (.+\.result\.json)$", out, re.M)
            res = json.loads(Path(m.group(1)).read_text()) if m else {}
            rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": rid,
                   "seed": seed, "launch": n, "mode": mode,
                   "returncode": proc.returncode, "status": res.get("status"),
                   "end_step": res.get("end_step"), "client_wall_s": time.time() - t0,
                   "app_id": (res.get("modal") or {}).get("app_id"),
                   "detail": res.get("detail"), "launch_log": str(launch_log),
                   "tail": out[-1500:] if not m else None}
            with a.log.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps({k: rec[k] for k in ("run_id", "launch", "mode", "status",
                                                   "end_step", "app_id")}), flush=True)
            if rec["status"] == "complete":
                break
            if rec["status"] != "interrupted":
                print(f"STOP: {rid} launch {n} ended {rec['status']!r} "
                      f"(rc {proc.returncode})", file=sys.stderr)
                return 1
        else:
            print(f"STOP: {rid} not complete after {MAX_LAUNCHES_PER_RUN} launches",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
