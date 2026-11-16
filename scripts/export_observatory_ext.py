"""The Observatory's v1.1 extension data (DECISIONS D-134), beside the Session 10 export.

    make obs-ext-select     # host: D-134 rules -> reports/session18/observatory/selection.json
    make obs-ext-export     # host snapshot, canonical container export (clean tree), snapshot
    make obs-ext-findings   # host: findings_ext.json copied from the three pinned results
    make web-data           # also runs web-data-ext: hash-checked copy into public/data/ext

``export`` re-derives each selected clip from its recorded MuJoCo states (body poses recomputed
and compared with the recorded ones, the simulator's own video rendered from the same states).
It never calls a policy, never clone-simulates and never steps the environment: ``reset`` only
compiles the start's model. It reads ``runs/`` and writes only the Session 18 tree. Suite names
come from ``selection.json``, which comes from the committed summaries.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
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

INDEX_SCHEMA = "roboforge.observatory_ext_index/1"
REGISTRY_SCHEMA = "roboforge.observatory_ext_registry/1"
PLATFORM_ID = "linux-aarch64-container-1f685638f28c"
SESSION10 = "reports/session10/observatory"


def write_gz(path: Path, obj: Any) -> None:
    """Deterministic gzipped JSON (mtime 0, sorted keys, no NaN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(data)


def cmd_select(out: Path) -> int:
    """Host: apply the D-134 rules and write ``selection.json``."""
    from roboforge.viz.obs_ext import build_selection

    doc = build_selection()
    doc.update(git=git_revision(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                               time.gmtime()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    missing = [c["role"] for c in doc["cases"] if c["state_id"] is None]
    print(f"{len(doc['clips'])} clips, {len(doc['cases'])} cases ({len(missing)} without a "
          f"start: {missing}); verification clips {doc['verification_clips']}")
    return 0


def cmd_findings(out: Path) -> int:
    """Host: ``findings_ext.json`` copied from the three pinned results."""
    from roboforge.viz.obs_ext import build_findings

    doc = build_findings()
    doc["git"] = git_revision()
    (out / "findings_ext.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"findings_ext.json from {len(doc['sources'])} pinned results")
    return 0


def _record_dirs(out: Path) -> list[str]:
    sel = json.loads((out / "selection.json").read_text())
    return sorted({c["records_dir"] for c in sel["clips"]})


def cmd_snapshot(out: Path, tag: str, compare: str | None) -> int:
    """Host: listing digest (path and size of every file) of each record directory read."""
    snap: dict[str, Any] = {}
    for rel in _record_dirs(out):
        root = REPO_ROOT / rel
        files = sorted(p for p in root.rglob("*") if p.is_file())
        listing = "\n".join(f"{p.relative_to(root).as_posix()} {p.stat().st_size}"
                            for p in files)
        snap[rel] = {"n_files": len(files),
                     "listing_sha256": hashlib.sha256(listing.encode()).hexdigest()}
    log = REPO_ROOT / "logs" / f"session{os.environ['ROBOFORGE_EVIDENCE_SESSION']}" \
        / "observatory_ext"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"isolation_{tag}.json").write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print(f"isolation snapshot {tag}: {len(snap)} record directories")
    if compare:
        before = json.loads((log / f"isolation_{compare}.json").read_text())
        if before != snap:
            print("ISOLATION FAIL: a record directory changed during the export")
            return 1
        print("isolation PASS: every record directory read is unchanged")
    return 0


def _check_record(clip: dict[str, Any], man: dict[str, Any], rec: Any, init: Any) -> None:
    ep = next(e for e in man["episodes"] if e["state_id"] == clip["state_id"])
    if ep["physics_sha256"] != rec.meta["physics_sha256"]:
        raise AssertionError(f"{clip['id']}: record physics hash differs from its manifest")
    if man["eval_run_id"] != clip["eval_run_id"]:
        raise AssertionError(f"{clip['id']}: manifest run differs from the selection")
    ref = rec.meta["episode"]["initial_state"]
    if (ref["state_sha256"], ref["model_xml_sha256"]) != (init.state_sha256,
                                                          init.model_xml_sha256):
        raise AssertionError(f"{clip['id']}: record start differs from the committed suite")
    if not np.array_equal(rec.arrays["state"][0], init.state):
        raise AssertionError(f"{clip['id']}: recorded frame 0 differs from the start state")
    if bool(rec.meta["outcome"]["success"]) != clip["success"]:
        raise AssertionError(f"{clip['id']}: record outcome differs from its summary")


def cmd_export(out: Path) -> int:  # noqa: PLR0915 - one linear pass, as Session 10's export
    """Container: every selected clip, its video and the verification reference frames."""
    from roboforge.eval.rollout import RolloutRecord
    from roboforge.sim import eval_states as es
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.sim.shadow_render import ShadowRenderer
    from roboforge.viz import obs_media as om
    from roboforge.viz.obs_scene import build_clip

    plat = evaluation_platform()
    if plat.get("platform_id") != PLATFORM_ID:
        raise SystemExit(f"export runs on the canonical container only, not {plat}")
    git = git_revision()
    if git.get("dirty"):
        raise SystemExit("export needs a clean tree (D-134)")
    sel = json.loads((out / "selection.json").read_text())
    starts: dict[str, Any] = {}
    by_suite: dict[tuple[str, str], set[str]] = {}
    for c in sel["clips"]:
        by_suite.setdefault((c["task"], c["suite"]), set()).add(c["state_id"])
    for (task, suite), ids in sorted(by_suite.items()):
        starts.update({s.state_id: s for s in es.load_suite(task, suite) if s.state_id in ids})
    envs = {t: RoboForgeEnv(t) for t in sorted({c["task"] for c in sel["clips"]})}
    files: list[str] = []
    entries, meshes = [], {}
    t_start = time.time()
    try:
        for i, c in enumerate(sel["clips"]):
            t0 = time.time()
            man = json.loads((REPO_ROOT / c["records_dir"] / "manifest.json").read_text())
            rec = RolloutRecord.load(REPO_ROOT / c["records_dir"] / "episodes" / c["state_id"])
            init = starts[c["state_id"]]
            _check_record(c, man, rec, init)
            env = envs[c["task"]]
            env.reset(init)  # compiles this start's model; no step is ever taken here
            kin, model = env.kinematics, env.model
            req = {"id": c["id"], "kind": f"{c['study']}_record", "roles": c["roles"],
                   "family": c["family"], "seed": c["seed"], "state_id": c["state_id"],
                   "partition": c["partition"], "records_dir": c["records_dir"]}
            clip = build_clip(req, rec.meta, rec.arrays, man, kin, model)
            clip["study"], clip["arm"] = c["study"], c["arm"]
            if c["display_label"]:
                clip["provenance"]["recorded_policy_label"] = clip["policy"]["label"]
                clip["policy"]["label"] = c["display_label"]
            for name, s in clip["meshes"].items():
                if name in meshes and meshes[name] != s:
                    raise AssertionError(f"mesh {name} differs between clips")
                meshes[name] = s
            states = rec.arrays["state"]
            shadow = ShadowRenderer(model)
            try:
                mp4 = out / "media" / f"{c['id']}_frontview.mp4"
                clip["media"] = {"video": {**om.render_video(shadow, states, mp4),
                                           "file": f"media/{mp4.name}"}}
                files.append(f"media/{mp4.name}")
                if c["id"] in sel["verification_clips"]:
                    t_last = len(states) - 1
                    fs = rec.meta["outcome"].get("first_success_step")
                    frames = sorted({0, t_last // 2, t_last} | ({fs} if fs else set()))
                    cats = dict(zip(rec.meta["bodies"]["names"],
                                    rec.meta["bodies"]["categories"], strict=True))
                    refs = om.reference_frames(shadow, kin, model, states, frames,
                                               out / "verification", c["id"], cats)
                    clip["reference_frames"] = refs
                    files += [f"verification/{r[k]}" for r in refs for k in ("rgb", "mask")]
            finally:
                shadow.close()
            clip["exported"] = {"platform": plat.get("platform_id"), "git": git}
            write_gz(out / "clips" / f"{c['id']}.json.gz", clip)
            files.append(f"clips/{c['id']}.json.gz")
            v = clip["verification"]["recomputed_vs_recorded"]
            entries.append({"id": c["id"], "study": c["study"], "arm": c["arm"],
                            "kind": req["kind"], "roles": c["roles"], "task": c["task"],
                            "family": c["family"], "seed": c["seed"],
                            "state_id": c["state_id"], "partition": c["partition"],
                            "suite": clip["suite"], "outcome": clip["outcome"],
                            "checkpoint": clip["checkpoint"], "eval_run_id": c["eval_run_id"],
                            "physics_sha256": rec.meta["physics_sha256"],
                            "arrays_sha256": rec.meta["arrays_sha256"],
                            "n_frames": clip["frames"]["n"],
                            "display_label": clip["policy"]["label"],
                            "has": sorted(k for k in ("reference_frames",) if k in clip),
                            "verification": v})
            print(f"[{i + 1}/{len(sel['clips'])}] {c['id']}: {clip['frames']['n']} frames, "
                  f"recompute |dpos| {v['body_pos_max_abs_m']:.1e} m, "
                  f"{time.time() - t0:.1f} s", flush=True)
    finally:
        for e in envs.values():
            e.close()
    (out / "meshes.json").write_text(json.dumps(
        {"schema": "roboforge.observatory_meshes/1", "platform": plat.get("platform_id"),
         "meshes": meshes}, indent=1, sort_keys=True) + "\n")
    files.append("meshes.json")
    index = {"schema": INDEX_SCHEMA, "decision": "D-134", "platform": plat, "git": git,
             "selection_sha256": sha256_file(out / "selection.json"),
             "wall_s": round(time.time() - t_start, 1), "clips": entries,
             "files": {f: sha256_file(out / f) for f in sorted(set(files))}}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(index['files'])} files in {index['wall_s']} s -> {repo_relative(out)}")
    return 0


def _copy_checked(src: Path, dst: Path, sha: str) -> None:
    if sha256_file(src) != sha:
        raise SystemExit(f"{repo_relative(src)}: SHA-256 differs from the index")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def mesh_problems(out: Path) -> list[str]:
    """The extension clips may use only meshes that Session 10 exported, with equal summaries
    (the site's meshes come from the Session 10 table, D-090)."""
    ext = json.loads((out / "meshes.json").read_text())["meshes"]
    s10 = json.loads((REPO_ROOT / SESSION10 / "meshes.json").read_text())["meshes"]
    return [f"mesh {k}: {'absent from' if k not in s10 else 'differs from'} the Session 10 "
            "export" for k, v in sorted(ext.items()) if s10.get(k) != v]


def cmd_web_data_ext(out: Path, web: Path) -> int:
    """Host: copy the Session 18 outputs into ``<web>/data/ext``, hash-checked; write the
    extension registry. Run after the Session 10 ``web-data`` step, which recreates ``data``."""
    problems = mesh_problems(out)
    if problems:
        raise SystemExit("\n".join(problems))
    index = json.loads((out / "index.json").read_text())
    data = web / "data" / "ext"
    if data.exists():
        shutil.rmtree(data)
    for rel, sha in index["files"].items():
        _copy_checked(out / rel, data / rel, sha)
    extra = {}
    for name in ("findings_ext.json", "selection.json", "viewer_verification.json"):
        if (out / name).exists():
            extra[name] = sha256_file(out / name)
            _copy_checked(out / name, data / name, extra[name])
    registry = {"schema": REGISTRY_SCHEMA, "decision": "D-134",
                "index_sha256": sha256_file(out / "index.json"),
                "source_dir": repo_relative(out), "clips": index["clips"],
                "files": {**index["files"], **extra},
                "export_platform": index["platform"], "export_git": index["git"]}
    (data / "ext_registry.json").write_text(json.dumps(registry, indent=1, sort_keys=True)
                                            + "\n")
    print(f"web data (extensions): {len(registry['files'])} files -> {repo_relative(data)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["select", "export", "findings", "snapshot",
                                       "web-data-ext"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--web-dir", type=Path, default=REPO_ROOT / "website" / "public")
    p.add_argument("--tag", default="before", help="snapshot: before or after")
    p.add_argument("--compare", default=None, help="snapshot: compare with this earlier tag")
    a = p.parse_args(argv)
    out = a.out_dir if a.out_dir.is_absolute() else REPO_ROOT / a.out_dir
    web = a.web_dir if a.web_dir.is_absolute() else REPO_ROOT / a.web_dir
    return {"select": lambda: cmd_select(out), "export": lambda: cmd_export(out),
            "findings": lambda: cmd_findings(out),
            "snapshot": lambda: cmd_snapshot(out, a.tag, a.compare),
            "web-data-ext": lambda: cmd_web_data_ext(out, web)}[a.command]()


if __name__ == "__main__":
    sys.exit(main())
