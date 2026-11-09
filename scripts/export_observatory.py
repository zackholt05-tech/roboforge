"""RoboForge Observatory data export (MASTER_SPEC §8.5; DECISIONS D-087-D-090).

    make observatory-select     # host: D-089 rules -> reports/session10/observatory/selection.json
    make observatory-export     # canonical container, clean tree: clips, media, denoise, clouds
    make observatory-findings   # host: findings.json from the frozen results.json files
    make web-data               # host: hash-checked copy into website/public/data (gitignored)
    make web-assets             # host: compiled visual meshes into website/public/assets (D-090)

``export`` reads only frozen evaluation records under ``runs/`` and the committed start
suites. Held-out records are recomputed and rendered from their recorded states, never stepped;
validation records are also replayed (denoising, clone-simulated candidate paths). Every output
file is listed with its SHA-256 in ``index.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file  # noqa: E402

INDEX_SCHEMA = "roboforge.observatory_index/1"
L2_OVERLAYS = "reports/session9/pc_feasibility/linux-aarch64-container-1f685638f28c"
BRANCH_WORKERS = 4  # D-069: at most 4 simulator processes in the canonical container


def write_gz(path: Path, obj: Any) -> None:
    """Deterministic gzipped JSON (mtime 0, sorted keys, no NaN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(data)


def cmd_select(out: Path) -> int:
    """Apply the D-089 rules and write ``selection.json``."""
    from roboforge.viz.obs_select import build_selection

    doc = build_selection()
    doc.update(git=git_revision(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                               time.gmtime()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{len(doc['clips'])} clips; verification clips {doc['verification_clips']}")
    return 0


def _initial_states(task: str, suite: str, ids: set[str]) -> dict[str, Any]:
    from roboforge.sim import eval_states as es

    return {s.state_id: s for s in es.load_suite(task, suite) if s.state_id in ids}


def _manifest(req: dict[str, Any]) -> dict[str, Any]:
    p = REPO_ROOT / req["records_dir"] / "manifest.json"
    if not p.exists():
        p = REPO_ROOT / "runs" / req["eval_run_id"] / "manifest.json"
    return json.loads(p.read_text())


def _check_record(req: dict[str, Any], man: dict[str, Any], rec: Any, init: Any) -> None:
    ep = next(e for e in man["episodes"] if e["state_id"] == req["state_id"])
    if ep["physics_sha256"] != rec.meta["physics_sha256"]:
        raise AssertionError(f"{req['id']}: record physics hash differs from its manifest")
    ref = rec.meta["episode"]["initial_state"]
    if (ref["state_sha256"], ref["model_xml_sha256"]) != (init.state_sha256,
                                                          init.model_xml_sha256):
        raise AssertionError(f"{req['id']}: record start differs from the committed suite")
    if not np.array_equal(rec.arrays["state"][0], init.state):
        raise AssertionError(f"{req['id']}: recorded frame 0 differs from the start state")


def _pc_spec(man: dict[str, Any]) -> Any:
    from roboforge.eval.pointcloud import load_pc_checkpoint
    from roboforge.train.pointcloud import PCConfig

    ck = man["checkpoint"]
    path = REPO_ROOT / "runs" / man["train_run_id"] / ck["path"]
    return PCConfig(load_pc_checkpoint(path, ck["sha256"])["config"]).cloud_spec


def _denoise(req: dict[str, Any], man: dict[str, Any], rec: Any, init: Any) -> dict[str, Any]:
    from roboforge.data.pipeline import load_specs
    from roboforge.eval.learned import LearnedPolicy
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.viz.obs_branch import run_pool
    from roboforge.viz.obs_denoise import DENOISE_SCHEMA, episode_traces
    from roboforge.viz.obs_scene import EEF_BODY

    ck, params = man["checkpoint"], man["policy"]["params"]
    policy = LearnedPolicy.from_checkpoint(REPO_ROOT / ck["path"], expected_sha256=ck["sha256"],
                                           policy_seed=params["policy_seed"],
                                           repeat=params["repeat"])
    _, action_spec = load_specs(req["task"])
    sim = rec.meta["simulator"]
    env = RoboForgeEnv(req["task"], horizon=sim["horizon"], hold_steps=sim["hold_steps"])
    env.close()

    def runner(jobs: list) -> list:
        return run_pool(req["task"], sim["horizon"], sim["hold_steps"], init,
                        rec.arrays["executed_action"], rec.arrays["state"], EEF_BODY,
                        BRANCH_WORKERS, jobs)

    eef = rec.meta["bodies"]["names"].index(EEF_BODY)
    tr = episode_traces(policy, rec, policy.codec(action_spec), runner, eef)
    return {"schema": DENOISE_SCHEMA, "clip_id": req["id"], "task": req["task"],
            "eval_run_id": man["eval_run_id"], "state_id": req["state_id"],
            "partition": init.partition, "checkpoint": ck, "policy": rec.meta["policy"],
            "outcome": rec.meta["outcome"],
            "horizons": {"T_o": policy.obs_horizon, "T_p": policy.pred_horizon,
                         "T_a": policy.exec_horizon},
            "kind": "export: real sampler calls replayed bitwise from a recorded validation "
                    "episode; not a new result", **tr}


def cmd_export(out: Path, only: list[str] | None = None,  # noqa: PLR0915 - one linear pass
               dev_dir: Path | None = None) -> int:
    """Container: every clip, its media, reference frames, clouds and denoising traces.

    ``dev_dir`` (with ``only``) is a development run: a dirty tree is allowed and everything is
    written to ``dev_dir`` instead (``logs/``), never to the evidence directory."""
    from roboforge.eval.rollout import RolloutRecord
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.sim.shadow_render import ShadowRenderer
    from roboforge.viz import obs_media as om
    from roboforge.viz.obs_scene import build_clip

    plat = evaluation_platform()
    if plat.get("platform_id") != "linux-aarch64-container-1f685638f28c":
        raise SystemExit(f"export runs on the canonical container only, not {plat}")
    git = git_revision()
    sel = json.loads((out / "selection.json").read_text())
    if dev_dir is not None:
        if not only:
            raise SystemExit("a development run needs --only")
        sel["clips"] = [c for c in sel["clips"] if c["id"] in only]
        out = dev_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "selection.json").write_text(json.dumps(sel, indent=1, sort_keys=True))
    elif git.get("dirty"):
        raise SystemExit("export needs a clean tree (D-089)")
    by_suite: dict[tuple[str, str], set[str]] = {}
    for c in sel["clips"]:
        suite = c["state_id"].rsplit("_", 1)[0].split("_", 1)[1]
        by_suite.setdefault((c["task"], suite), set()).add(c["state_id"])
    starts: dict[str, Any] = {}
    for (task, suite), ids in sorted(by_suite.items()):
        starts.update(_initial_states(task, suite, ids))
    envs = {t: RoboForgeEnv(t) for t in sorted({c["task"] for c in sel["clips"]})}
    files: list[str] = []
    entries, meshes = [], {}
    t_start = time.time()
    try:
        for i, req in enumerate(sel["clips"]):
            t0 = time.time()
            man = _manifest(req)
            rec = RolloutRecord.load(REPO_ROOT / req["records_dir"] / "episodes" /
                                     req["state_id"])
            init = starts[req["state_id"]]
            _check_record(req, man, rec, init)
            env = envs[req["task"]]
            env.reset(init)  # compiles this start's model; no step is ever taken here
            kin, model = env.kinematics, env.model
            clip = build_clip(req, rec.meta, rec.arrays, man, kin, model)
            for name, s in clip["meshes"].items():
                if name in meshes and meshes[name] != s:
                    raise AssertionError(f"mesh {name} differs between clips")
                meshes[name] = s
            states = rec.arrays["state"]
            shadow = ShadowRenderer(model)
            try:
                mp4 = out / "media" / f"{req['id']}_frontview.mp4"
                clip["media"] = {"video": {**om.render_video(shadow, states, mp4),
                                           "file": f"media/{mp4.name}"}}
                files.append(f"media/{mp4.name}")
                if req["id"] in sel["verification_clips"]:
                    t_last = len(states) - 1
                    fs = rec.meta["outcome"].get("first_success_step")
                    frames = sorted({0, t_last // 2, t_last} | ({fs} if fs else set()))
                    cats = dict(zip(rec.meta["bodies"]["names"],
                                    rec.meta["bodies"]["categories"], strict=True))
                    refs = om.reference_frames(shadow, kin, model, states, frames,
                                               out / "verification", req["id"], cats)
                    clip["reference_frames"] = refs
                    files += [f"verification/{r[k]}" for r in refs for k in ("rgb", "mask")]
                dframes = om.decision_frame_indices(rec.arrays)
                if req["family"] == "rgb_diffusion":
                    pv = om.policy_rgb_frames(shadow, states, dframes, out / "policy_views",
                                              req["id"])
                    clip["policy_views"] = pv
                    files += [f"policy_views/{f}" for v in pv["files"].values() for f in v]
            finally:
                shadow.close()
            if req["family"] == "pc_diffusion":
                clouds = om.policy_clouds(model, _pc_spec(man), states, dframes)
                write_gz(out / "clouds" / f"{req['id']}.json.gz",
                         {"schema": "roboforge.observatory_clouds/1", "clip_id": req["id"],
                          **clouds})
                clip["clouds"] = f"clouds/{req['id']}.json.gz"
                files.append(clip["clouds"])
            if req["kind"] == "validation_denoising":
                den = _denoise(req, man, rec, init)
                write_gz(out / "denoise" / f"{req['id']}.json.gz", den)
                clip["denoise"] = f"denoise/{req['id']}.json.gz"
                files.append(clip["denoise"])
            clip["exported"] = {"platform": plat.get("platform_id"), "git": git}
            write_gz(out / "clips" / f"{req['id']}.json.gz", clip)
            files.append(f"clips/{req['id']}.json.gz")
            v = clip["verification"]["recomputed_vs_recorded"]
            entries.append({"id": req["id"], "kind": req["kind"], "roles": req["roles"],
                            "task": req["task"], "family": req["family"], "seed": req["seed"],
                            "state_id": req["state_id"], "partition": req["partition"],
                            "suite": clip["suite"], "outcome": clip["outcome"],
                            "checkpoint": clip["checkpoint"],
                            "eval_run_id": man["eval_run_id"],
                            "physics_sha256": rec.meta["physics_sha256"],
                            "arrays_sha256": rec.meta["arrays_sha256"],
                            "n_frames": clip["frames"]["n"],
                            "has": sorted(k for k in ("denoise", "clouds", "policy_views",
                                                      "reference_frames") if k in clip),
                            "verification": v})
            print(f"[{i + 1}/{len(sel['clips'])}] {req['id']}: {clip['frames']['n']} frames, "
                  f"recompute |dpos| {v['body_pos_max_abs_m']:.1e} m, "
                  f"{time.time() - t0:.1f} s", flush=True)
    finally:
        for e in envs.values():
            e.close()
    (out / "meshes.json").write_text(json.dumps(
        {"schema": "roboforge.observatory_meshes/1", "platform": plat.get("platform_id"),
         "meshes": meshes}, indent=1, sort_keys=True) + "\n")
    files.append("meshes.json")
    index = {"schema": INDEX_SCHEMA, "platform": plat, "git": git,
             "selection_sha256": sha256_file(out / "selection.json"),
             "wall_s": round(time.time() - t_start, 1), "clips": entries,
             "files": {f: sha256_file(out / f) for f in sorted(set(files))}}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(index['files'])} files in {index['wall_s']} s -> {repo_relative(out)}")
    return 0


def cmd_findings(out: Path) -> int:
    """Host: ``findings.json`` from the frozen results (see ``roboforge.viz.obs_findings``)."""
    from roboforge.viz.obs_findings import build_findings

    sel = json.loads((out / "selection.json").read_text())
    doc = build_findings(sel)
    doc["git"] = git_revision()
    (out / "findings.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"findings.json from {len(doc['sources'])} frozen sources")
    return 0


def _copy_checked(src: Path, dst: Path, sha: str) -> None:
    if sha256_file(src) != sha:
        raise SystemExit(f"{repo_relative(src)}: SHA-256 differs from the index")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def cmd_web_data(out: Path, web: Path) -> int:
    """Host: copy the committed outputs into ``<web>/data``, hash-checked; write the registry."""
    index = json.loads((out / "index.json").read_text())
    data = web / "data"
    if data.exists():
        shutil.rmtree(data)
    for rel, sha in index["files"].items():
        _copy_checked(out / rel, data / rel, sha)
    extra = {}
    for name in ("findings.json", "selection.json", "viewer_verification.json"):
        if (out / name).exists():
            extra[name] = sha256_file(out / name)
            _copy_checked(out / name, data / name, extra[name])
    overlays = sorted((REPO_ROOT / L2_OVERLAYS).glob("lift_overlay_*.png"))
    for p in overlays:
        _copy_checked(p, data / "overlays" / p.name, sha256_file(p))
    registry = {"schema": "roboforge.observatory_registry/1",
                "index_sha256": sha256_file(out / "index.json"),
                "source_dir": repo_relative(out), "clips": index["clips"],
                "files": {**index["files"], **extra,
                          **{f"overlays/{p.name}": sha256_file(p) for p in overlays}},
                "overlay_source": L2_OVERLAYS, "export_platform": index["platform"],
                "export_git": index["git"]}
    (data / "registry.json").write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    print(f"web data: {len(registry['files'])} files -> {repo_relative(data)}")
    return 0


def _mesh_arrays(model: Any, m: int) -> tuple[np.ndarray, np.ndarray]:
    a, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
    fa, fn = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
    return (np.array(model.mesh_vert[a:a + n], dtype=np.float64),
            np.ascontiguousarray(model.mesh_face[fa:fa + fn], dtype="<i4"))


def cmd_web_assets(out: Path, web: Path) -> int:
    """Host: quantized meshes of the pinned robosuite, checked against the container's summary."""
    import mujoco

    from roboforge.sim import eval_states as es
    from roboforge.sim.env_adapter import RoboForgeEnv

    want = json.loads((out / "meshes.json").read_text())["meshes"]
    got: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in ("lift", "can"):
        env = RoboForgeEnv(task)
        try:
            env.reset(es.load_suite(task, "validation")[0])
            model = env.model
            for m in range(model.nmesh):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m)
                if name in want and name not in got:
                    got[name] = _mesh_arrays(model, m)
        finally:
            env.close()
    missing = sorted(set(want) - set(got))
    if missing:
        raise SystemExit(f"meshes not found in the host models: {missing}")
    blobs, table, worst = [], {}, 0.0
    offset = 0
    for name in sorted(want):
        v, f = got[name]
        w = want[name]
        if hashlib.sha256(f.tobytes()).hexdigest() != w["face_sha256"] or len(v) != w["n_vert"]:
            raise SystemExit(f"{name}: faces/vertex count differ from the container export")
        for key, val in (("vert_min", v.min(0)), ("vert_max", v.max(0)),
                         ("vert_mean", v.mean(0))):
            worst = max(worst, float(np.abs(np.asarray(w[key]) - val).max()))
        lo, hi = v.min(0), v.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        q = np.round((v - lo) / span * 65535).astype("<u2")
        idx = f.astype("<u2" if len(v) < 65536 else "<u4")
        pos_bytes, idx_bytes = q.tobytes(), idx.tobytes()
        pad = (-len(pos_bytes)) % 4
        table[name] = {"n_vert": len(v), "n_face": len(f), "lo": lo.tolist(),
                       "span": span.tolist(), "pos_offset": offset,
                       "index_offset": offset + len(pos_bytes) + pad,
                       "index_type": "uint16" if len(v) < 65536 else "uint32",
                       "max_quantization_error_m": float((span / 65535 / 2).max())}
        blobs += [pos_bytes, b"\0" * pad, idx_bytes, b"\0" * ((-len(idx_bytes)) % 4)]
        offset += len(pos_bytes) + pad + len(idx_bytes) + (-len(idx_bytes)) % 4
    if worst > 1e-9:
        raise SystemExit(f"host meshes differ from the container's by {worst:.2e} m")
    dest = web / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    blob = b"".join(blobs)
    (dest / "meshes.bin").write_bytes(blob)
    (dest / "meshes.json").write_text(json.dumps({
        "schema": "roboforge.web_meshes/1", "quantization": "uint16 per-mesh bounding box",
        "source": "compiled MuJoCo meshes of the pinned local robosuite 1.5.1 install (MIT); "
                  "extracted at build time, never committed (D-090)",
        "host_vs_container_max_abs_m": worst, "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(), "meshes": table}, indent=1) + "\n")
    print(f"{len(table)} meshes, {len(blob) / 1e6:.2f} MB, host vs container {worst:.1e} m")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["select", "export", "findings", "web-data",
                                       "web-assets"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--web-dir", type=Path, default=REPO_ROOT / "website" / "public")
    p.add_argument("--only", nargs="+", default=None, help="development: these clip IDs only")
    p.add_argument("--dev-dir", type=Path, default=None,
                   help="development output directory (logs/...); allows a dirty tree")
    a = p.parse_args(argv)
    out = a.out_dir if a.out_dir.is_absolute() else REPO_ROOT / a.out_dir
    web = a.web_dir if a.web_dir.is_absolute() else REPO_ROOT / a.web_dir
    dev = REPO_ROOT / a.dev_dir if a.dev_dir else None
    return {"select": lambda: cmd_select(out), "export": lambda: cmd_export(out, a.only, dev),
            "findings": lambda: cmd_findings(out),
            "web-data": lambda: cmd_web_data(out, web),
            "web-assets": lambda: cmd_web_assets(out, web)}[a.command]()


if __name__ == "__main__":
    sys.exit(main())
