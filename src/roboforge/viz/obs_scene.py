"""Observatory clips: the recorded episode as web data, re-derived from its saved states (D-089).

For one frozen episode record this module builds:

* the **scene table**: every visual geom (group 1) of every non-world body, with its geometry
  type, size, pose *relative to its body* (``geom_pos``, ``geom_quat``) and a colour role, plus
  a summary of each mesh it uses (vertex/face counts, face hash, vertex statistics). Meshes are
  never written here (D-090); ``make web-assets`` extracts them on the host and checks them
  against this summary;
* the **frames**: world position and ``(w, x, y, z)`` quaternion of every exported body at every
  recorded frame, recomputed by ``mj_kinematics`` on scratch data from the recorded flattened
  states (:class:`~roboforge.sim.kinematics.KinematicsExtractor`) and required to equal the
  record's own ``body_pos`` / ``body_quat_wxyz``;
* a **geom composition check**: body pose ∘ local geom pose must equal MuJoCo's ``geom_xpos`` /
  ``geom_xmat`` (this is exactly what the browser composes);
* the static **cameras** (``frontview``, ``agentview``, ``birdview``, ``sideview``) as world
  poses and vertical fields of view, so the viewer can reproduce MuJoCo's views.

Nothing here steps physics. A held-out record is only ever loaded, recomputed and rendered.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from roboforge.viz import coords

CLIP_SCHEMA = "roboforge.observatory_clip/1"
SCENE_CAMERAS = ("frontview", "agentview", "birdview", "sideview")
GEOM_KIND = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
EEF_BODY = "gripper0_right_eef"
ROUND = 6  # decimals for exported positions (1 µm) and quaternion components
LABEL_CLIP = ("recorded closed-loop rollout: an illustration of a frozen evaluation, drawn from "
              "the recorded MuJoCo states")


def _name(model: Any, obj: Any, i: int) -> str | None:
    import mujoco

    return mujoco.mj_id2name(model, obj, i)


def colour_role(model: Any, g: int, category: str, body: str) -> str:
    """A palette role for a geom: the site restyles colours, never geometry (D-089 labels)."""
    mat = int(model.geom_matid[g])
    rgba = np.array(model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g])
    if category == "free_object":
        return "object"
    if body.startswith("Visual") or "visual" in body.lower() and category == "fixture":
        return "target"
    if category == "fixture":
        return "table" if body == "table" else "fixture"
    if body.startswith("fixed_mount"):
        return "pedestal"
    lum = float(rgba[:3].mean())
    if rgba[2] > 0.6 and rgba[0] < 0.2:
        return "robot_accent"
    if (rgba[1] > 0.9 and rgba[0] < 0.1) or (rgba[0] > 0.9 and rgba[1] < 0.1):
        return "robot_accent"
    return "robot_light" if lum > 0.6 else "robot_dark"


def mesh_summary(model: Any, m: int) -> dict[str, Any]:
    """Counts, exact face hash and vertex statistics of compiled mesh ``m``."""
    a, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
    fa, fn = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
    v = np.array(model.mesh_vert[a:a + n], dtype=np.float64)
    f = np.ascontiguousarray(model.mesh_face[fa:fa + fn], dtype="<i4")
    return {"name": _name(model, _mesh_obj(), m), "n_vert": n, "n_face": fn,
            "face_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "vert_min": v.min(0).tolist(), "vert_max": v.max(0).tolist(),
            "vert_mean": v.mean(0).tolist()}


def _mesh_obj() -> Any:
    import mujoco

    return mujoco.mjtObj.mjOBJ_MESH


def scene_table(model: Any, body_names: list[str], categories: list[str]
                ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Visual geoms of the exported bodies and the summary of every mesh they use."""
    import mujoco

    ob = mujoco.mjtObj
    cat = dict(zip(body_names, categories, strict=True))
    geoms, meshes = [], {}
    for g in range(model.ngeom):
        body = _name(model, ob.mjOBJ_BODY, int(model.geom_bodyid[g]))
        t = int(model.geom_type[g])
        if body not in cat or int(model.geom_group[g]) != 1 or t not in GEOM_KIND:
            continue
        mat = int(model.geom_matid[g])
        rgba = (model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]).tolist()
        rec = {"body": body, "kind": GEOM_KIND[t], "name": _name(model, ob.mjOBJ_GEOM, g),
               "size": [float(x) for x in model.geom_size[g]],
               "pos": [float(x) for x in model.geom_pos[g]],
               "quat_wxyz": [float(x) for x in model.geom_quat[g]],
               "rgba": [round(float(x), 4) for x in rgba],
               "role": colour_role(model, g, cat[body], body)}
        if t == 7:
            m = int(model.geom_dataid[g])
            s = mesh_summary(model, m)
            meshes[s["name"]] = s
            rec["mesh"] = s["name"]
        geoms.append(rec)
    return geoms, meshes


def recompute_frames(kin: Any, states: np.ndarray, record_arrays: dict[str, np.ndarray]
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Body poses from the recorded states, and their difference from the recorded poses."""
    pos, quat = [], []
    for s in states:
        f = kin.frame(s)
        pos.append(f["body_pos"])
        quat.append(f["body_quat_wxyz"])
    pos, quat = np.asarray(pos), np.asarray(quat)
    dq = np.minimum(np.abs(quat - record_arrays["body_quat_wxyz"]).max(-1),
                    np.abs(quat + record_arrays["body_quat_wxyz"]).max(-1))
    err = {"body_pos_max_abs_m": float(np.abs(pos - record_arrays["body_pos"]).max()),
           "body_quat_max_abs": float(dq.max())}
    return pos, quat, err


def geom_composition_error(model: Any, kin: Any, states: np.ndarray, body_names: list[str],
                           geoms: list[dict[str, Any]], frames: list[int]) -> dict[str, float]:
    """Max |body ∘ local − MuJoCo| over the listed frames for every exported geom."""
    import mujoco

    ob = mujoco.mjtObj
    data = mujoco.MjData(model)
    bidx = {n: i for i, n in enumerate(body_names)}
    gid = [mujoco.mj_name2id(model, ob.mjOBJ_GEOM, g["name"]) if g["name"] else None
           for g in geoms]
    ep = er = 0.0
    for k in frames:
        s = states[k]
        data.qpos[:] = s[1:1 + model.nq]
        mujoco.mj_kinematics(model, data)
        f = kin.frame(s)
        for g, i in zip(geoms, gid, strict=True):
            if i is None or i < 0:
                continue
            bp, bq = f["body_pos"][bidx[g["body"]]], f["body_quat_wxyz"][bidx[g["body"]]]
            wp = bp + coords.quat_to_mat(bq) @ np.asarray(g["pos"])
            wr = coords.quat_to_mat(coords.quat_mul(bq, np.asarray(g["quat_wxyz"])))
            ep = max(ep, float(np.abs(wp - data.geom_xpos[i]).max()))
            er = max(er, float(np.abs(wr - data.geom_xmat[i].reshape(3, 3)).max()))
    return {"geom_pos_max_abs_m": ep, "geom_rot_max_abs": er}


def cameras(kin: Any, state: np.ndarray, names: tuple[str, ...] = SCENE_CAMERAS
            ) -> dict[str, dict[str, Any]]:
    """World pose (MuJoCo frame, ``(w,x,y,z)``) and vertical FOV of the named cameras."""
    out = {}
    for c in names:
        if c not in kin.camera_names:
            continue
        cam = kin.camera(state, c, 1, 1)
        out[c] = {"pos": [float(x) for x in cam["pos"]],
                  "quat_wxyz": [float(x) for x in coords.mat_to_quat(cam["rot_world_from_cam"])],
                  "fovy_deg": float(cam["fovy_deg"])}
    return out


def success_rule(model: Any, state: np.ndarray, task: str) -> dict[str, Any]:
    """The task's success predicate as documented in docs/SIMULATOR_CONTRACT.md."""
    if task != "lift":
        return {"text": "the can rests inside the goal bin (robosuite PickPlace._check_success)",
                "source": "docs/SIMULATOR_CONTRACT.md"}
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[:] = state[1:1 + model.nq]
    mujoco.mj_kinematics(model, d)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    top = float(d.geom_xpos[g][2] + model.geom_size[g][2])
    return {"text": "cube body z > table top + 0.04 m", "source": "docs/SIMULATOR_CONTRACT.md",
            "body": "cube_main", "table_top_z_m": top, "z_threshold_m": top + 0.04}


def _r(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def require_provenance(meta: dict[str, Any], manifest: dict[str, Any]) -> None:
    """MASTER_SPEC §8.5: refuse a clip without checkpoint hash, run provenance or build."""
    need = {
        "physics_sha256": meta.get("physics_sha256"),
        "arrays_sha256": meta.get("arrays_sha256"),
        "policy.autonomous": meta.get("policy", {}).get("autonomous") is not None,
        # A checkpoint hash is required unless the policy is non-learned (the random baseline,
        # whose manifest has ``checkpoint: null``).
        "checkpoint_sha256": (manifest.get("checkpoint") or {}).get("sha256")
        or meta.get("policy", {}).get("learned") is False,
        "platform_id": meta.get("platform", {}).get("platform_id"),
        "simulator.packages": meta.get("simulator", {}).get("packages"),
        "initial_state.model_xml_sha256": meta.get("episode", {}).get("initial_state", {})
        .get("model_xml_sha256"),
        "eval_run_id": manifest.get("eval_run_id")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise ValueError(f"record lacks provenance: {missing}")


def build_clip(request: dict[str, Any], meta: dict[str, Any], arrays: dict[str, np.ndarray],
               manifest: dict[str, Any], kin: Any, model: Any) -> dict[str, Any]:
    """The clip document of one record (see the module docstring)."""
    require_provenance(meta, manifest)
    states = arrays["state"]
    names = list(meta["bodies"]["names"])
    cats = list(meta["bodies"]["categories"])
    if names != kin.body_names:
        raise AssertionError("record bodies differ from the compiled model's bodies")
    pos, quat, err = recompute_frames(kin, states, arrays)
    geoms, meshes = scene_table(model, names, cats)
    t_last = len(states) - 1
    check_frames = sorted({0, t_last // 2, t_last})
    err.update(geom_composition_error(model, kin, states, names, geoms, check_frames))
    dynamic = [i for i, c in enumerate(cats) if c != "fixture"]
    ck = manifest.get("checkpoint") or {}
    dec = arrays.get("decision_index")
    off = arrays.get("chunk_offset")
    lat = arrays.get("planning_latency_ms")
    out = {
        "schema": CLIP_SCHEMA, "id": request["id"], "label": LABEL_CLIP,
        "kind": request["kind"], "roles": request["roles"], "task": meta["task"],
        "family": request["family"], "seed": request["seed"],
        "state_id": request["state_id"], "partition": request["partition"],
        "suite": meta["episode"]["initial_state"]["source"].get("suite"),
        "policy": {k: meta["policy"].get(k) for k in ("label", "name", "kind", "autonomous",
                                                     "learned")},
        "checkpoint": {"sha256": ck.get("sha256"), "step": ck.get("step"),
                       "train_run_id": manifest.get("train_run_id")},
        "outcome": meta["outcome"],
        "provenance": {"eval_run_id": manifest["eval_run_id"],
                       "record": f"{request['records_dir']}/episodes/{request['state_id']}",
                       "physics_sha256": meta["physics_sha256"],
                       "arrays_sha256": meta["arrays_sha256"],
                       "platform": meta["platform"].get("platform_id"),
                       "eval_git": manifest.get("git"), "simulator": meta["simulator"],
                       "initial_state": meta["episode"]["initial_state"],
                       "frame_convention": meta["frame_convention"],
                       "quaternion_convention": meta["quaternion_convention"]},
        "control_dt_s": float(meta["simulator"]["control_dt_s"]),
        "bodies": [{"name": n, "category": c, "dynamic": i in dynamic}
                   for i, (n, c) in enumerate(zip(names, cats, strict=True))],
        "eef_body": EEF_BODY,
        "geoms": geoms, "meshes": meshes, "cameras": cameras(kin, states[0]),
        "frames": {"n": int(len(states)), "time_s": _r(arrays["time_s"]),
                   "success": [bool(x) for x in arrays["success"]],
                   "static_pos": _r(pos[0]), "static_quat_wxyz": _r(quat[0]),
                   "pos": _r(pos[:, dynamic]), "quat_wxyz": _r(quat[:, dynamic]),
                   "dynamic_index": dynamic,
                   "joint_qpos": _r(arrays["joint_qpos"])},
        "joints": list(meta["joints"]),
        "hierarchy": [dict(b.__dict__) for b in kin.hierarchy()],
        "success_rule": success_rule(model, states[0], meta["task"]),
        "actions": {"names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
                    "executed": _r(arrays["executed_action"]),
                    "clipped": [bool(x) for x in arrays["clip_mask"].any(axis=1)],
                    "decision_index": [] if dec is None else [int(x) for x in dec],
                    "chunk_offset": [] if off is None else [int(x) for x in off],
                    "latency_ms": [] if lat is None else [None if not np.isfinite(x) else
                                                          round(float(x), 3) for x in lat]},
        "verification": {"recomputed_vs_recorded": err,
                         "geom_check_frames": check_frames,
                         "rounding_decimals": ROUND},
    }
    return out
