"""Session 12 (D-098): hash-check one frozen checkpoint against the Level 1 freeze and load it
strictly with the fresh clone's code. No rollout, no policy evaluation."""
import json, sys
from pathlib import Path
from roboforge.eval import freeze as fz
from roboforge.eval.learned import model_from_payload
from roboforge.provenance import sha256_file
from roboforge.train.checkpoint import load_checkpoint

src = Path(sys.argv[1])
frozen = fz.load_freeze()
entry = next(e for e in fz.entries(frozen, "can") if e.get("train_run_id") == "s7_diffusion_unet_cos_can_seed0_99251da3")
ck = src / entry["train_run_id"] / "checkpoints" / f"step_{int(entry['step']):07d}.pt"
sha = sha256_file(ck)
payload = load_checkpoint(ck, expected_sha256=entry["checkpoint_sha256"])
model = model_from_payload(payload)
n = sum(p.numel() for p in model.parameters())
print(json.dumps({"entry": {k: entry[k] for k in ("task", "family", "train_run_id", "step", "checkpoint_sha256")},
                  "file_sha256": sha, "matches_freeze": sha == entry["checkpoint_sha256"],
                  "loaded_class": type(model).__name__, "parameters": n}, indent=1))
