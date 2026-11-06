"""Dump eval-start 0 (validation) as generated on this platform, for cross-platform diffing."""
import json, sys
from roboforge.sim.backend import configure_mujoco_gl
configure_mujoco_gl()
import numpy as np
from roboforge.sim.factory import load_sim_config, make_env
from roboforge.sim.snapshot import generate
task, out = sys.argv[1], sys.argv[2]
env = make_env(load_sim_config(task))
recs = []
for seed in (100000, 100001, 100002):
    init = generate(env, task, seed, f"s{seed}", "eval_validation")
    recs.append({"seed": seed, "xml_sha": init.model_xml_sha256, "state_sha": init.state_sha256,
                 "state": init.state.tolist(), "xml": init.model_xml})
json.dump(recs, open(out, "w"))
print("wrote", out)
