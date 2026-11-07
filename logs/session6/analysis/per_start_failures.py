"""Per-start failure counts over every Session 6 checkpoint evaluation, by family (validation
suite only; reads summaries). Used for KI-028 (is lift_validation_0029 diffusion-specific?).

    python3 logs/session6/analysis/per_start_failures.py > logs/session6/analysis/per_start_failures.log
"""
import collections
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[3]
FAM = ("bc_mlp_chunk", "bc_mlp", "bc_rnn", "diffusion_unet")
fails = collections.defaultdict(collections.Counter)
n_eval, n_ep, n_fail = collections.Counter(), collections.Counter(), collections.Counter()
for f in sorted((root / "reports/session6/validation").glob("s6_*.json")):
    s = json.loads(f.read_text())
    rid = s["train_run_id"][3:]
    fam = next(x for x in FAM if rid.startswith(x + "_"))
    n_eval[fam] += 1
    for e in s["episodes"]:
        n_ep[fam] += 1
        if not e["success"]:
            n_fail[fam] += 1
            fails[fam][e["state_id"]] += 1
out = {}
for fam in sorted(n_eval):
    top = fails[fam].most_common(6)
    out[fam] = {"evaluations": n_eval[fam], "episodes": n_ep[fam], "failures": n_fail[fam],
                "distinct_failing_starts": len(fails[fam]), "top_failing_starts": top}
    print(fam, out[fam])
print("lift_validation_0029 failures by family:",
      {f: fails[f]["lift_validation_0029"] for f in sorted(n_eval)})
(root / "logs/session6/analysis/per_start_failures.json").write_text(json.dumps(out, indent=1) + "\n")
