"""Export one recorded rollout as a small web clip and verify its coordinates (Session 2).

The clip is the per-frame body poses of a real :class:`~roboforge.eval.rollout.RolloutRecord`,
in both MuJoCo and three.js frames, plus the link hierarchy, joint values, fixed cameras and
verification points. The checks, all against the physics engine:

1. **Forward kinematics:** the exported hierarchy plus joint values reproduce MuJoCo's body
   poses at every frame (:func:`roboforge.viz.coords.forward_kinematics`).
2. **Conversion:** the three.js-convention projection of converted points equals the
   MuJoCo-frame projection.
3. **Reference renders:** at ≥ 3 timestamps and 2 fixed cameras, every pixel of each tracked
   visual geom's MuJoCo segmentation mask must lie inside the projected bounding box of the
   geom's vertices (dilated by 1 px). Occlusion can only shrink a mask, so this holds for
   every geom. For the task object at t = 0, which the robot does not occlude, the mask box and
   the projected box must also agree edge by edge within 1 px.

The clip states the policy's ``autonomous`` flag, the record's ``physics_sha256`` and the
snapshot hashes. A clip without them is refused.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.viz import coords

CLIP_SCHEMA = "roboforge.web_clip/1"
GEOM_TYPES = {"box": 6, "mesh": 7}  # mujoco.mjtGeom


def geom_vertices(model: Any, data: Any, g: int) -> np.ndarray | None:
    """World-frame vertices of a box or mesh geom (None for other types)."""
    pos, rot = np.array(data.geom_xpos[g]), np.array(data.geom_xmat[g]).reshape(3, 3)
    t = int(model.geom_type[g])
    if t == GEOM_TYPES["box"]:
        s = np.array(model.geom_size[g])
        local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)]) * s
    elif t == GEOM_TYPES["mesh"]:
        m = int(model.geom_dataid[g])
        a, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        local = np.array(model.mesh_vert[a:a + n])
    else:
        return None
    return pos + local @ rot.T


def tracked_geoms(model: Any, body_names: list[str]) -> list[int]:
    """Visual (group 1) box/mesh geoms attached to the named bodies."""
    import mujoco

    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names} - {-1}
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids
            and int(model.geom_group[g]) == 1 and int(model.geom_type[g]) in GEOM_TYPES.values()]


def check_geoms_against_render(model: Any, data: Any, seg: np.ndarray, cam: dict[str, Any],
                               geoms: list[int]) -> list[dict[str, Any]]:
    """Projected vertex bounding box vs segmentation mask for each geom (one image)."""
    import mujoco

    h, w = seg.shape[:2]
    out = []
    for g in geoms:
        v = geom_vertices(model, data, g)
        px = coords.project_mujoco(v, cam["pos"], cam["rot_world_from_cam"], cam["fovy_deg"],
                                   w, h)
        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] == g)
        rec: dict[str, Any] = {"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                               "mask_pixels": int(mask.sum())}
        lo, hi = px.min(0), px.max(0)
        rec["projected_bbox"] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        if mask.any():
            ii, jj = np.nonzero(mask)
            cu, cv = jj + 0.5, ii + 0.5
            inside = ((cu >= lo[0] - 1) & (cu <= hi[0] + 1) & (cv >= lo[1] - 1)
                      & (cv <= hi[1] + 1))
            mb = [float(jj.min()), float(ii.min()), float(jj.max() + 1), float(ii.max() + 1)]
            in_frame = lo[0] >= 0 and lo[1] >= 0 and hi[0] <= w and hi[1] <= h
            rec.update(mask_bbox=mb, containment=float(inside.mean()),
                       bbox_edge_err_px=float(np.max(np.abs(np.array(mb) -
                                                            np.array(rec["projected_bbox"])))),
                       projected_inside_image=bool(in_frame))
        out.append(rec)
    return out


def require_provenance(meta: dict[str, Any]) -> None:
    """Refuse to export a record without policy flags, physics hash and snapshot hashes."""
    need = (meta.get("physics_sha256"), meta.get("policy", {}).get("autonomous") is not None,
            meta.get("episode", {}).get("initial_state", {}).get("state_sha256"),
            meta.get("episode", {}).get("initial_state", {}).get("model_xml_sha256"),
            meta.get("platform", {}).get("platform_id"))
    if not all(need):
        raise ValueError("record lacks provenance (policy flags / hashes / platform)")
