"""Session 5 prerequisite: the Session 3-4 pilot training runs exist, are complete, and every
checkpoint's SHA-256 matches its manifest (read-only)."""
import hashlib, json, sys
from pathlib import Path

RUNS = ["s3_bc_mlp_lift_seed0_dbf312cd", "s3_bc_rnn_lift_seed0_2e7cffe3",
        "s3_bc_mlp_chunk_lift_seed0_bb8b4dd6", "s4_diffusion_unet_lift_seed0_336e15e9"]
ok = True
for rid in RUNS:
    d = Path("runs") / rid
    m = json.loads((d / "manifest.json").read_text())
    bad = []
    for c in m["checkpoints"]:
        h = hashlib.sha256(Path(c["path"]).read_bytes()).hexdigest()
        if h != c["sha256"]:
            bad.append(c["step"])
    status = m["status"] == "complete" and not bad and m["git"]["dirty"] is False
    ok &= status
    print(f"{rid}: status={m['status']} git={m['git']['sha'][:7]} dirty={m['git']['dirty']} "
          f"checkpoints={len(m['checkpoints'])} hash_mismatch={bad} "
          f"wall_s={m['timing']['wall_s']:.0f} -> {'PASS' if status else 'FAIL'}")
print("ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
