"""Diagnostic: ShadowRenderer (RoboForgeEnv.render) vs robosuite's renderer, every camera, 3 validation starts per task."""
import json, sys
import numpy as np
from roboforge.provenance import environment_snapshot, evaluation_platform
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.sim.factory import load_sim_config, make_env
from roboforge.sim.renderer import render_rgb
from roboforge.sim.snapshot import restore

out = {"what": "max |shadow - robosuite| grey level and #pixels differing, 256x256", "environment": environment_snapshot(),
       "platform": evaluation_platform(), "results": {}}
for task in ("lift", "can"):
    env = RoboForgeEnv(task)
    rs = make_env(load_sim_config(task), offscreen=True)
    res = {}
    for init in load_suite(task, "validation")[:3]:
        env.reset(init); restore(rs, init)
        names = [rs.sim.model.camera_id2name(i) for i in range(rs.sim.model.ncam)]
        for cam in names:
            a = env.render(cam, 256, 256).astype(int); b = render_rgb(rs, cam, 256, 256).astype(int)
            d = np.abs(a - b)
            res[f"{init.state_id}/{cam}"] = {"max_abs": int(d.max()), "n_pixels_differing": int((d.max(axis=2) > 0).sum())}
    out["results"][task] = res
    out[f"{task}_summary"] = {"max_abs": max(v["max_abs"] for v in res.values()),
                              "max_pixels_differing": max(v["n_pixels_differing"] for v in res.values())}
    rs.close(); env.close()
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print({k: v for k, v in out.items() if k.endswith("summary")})
