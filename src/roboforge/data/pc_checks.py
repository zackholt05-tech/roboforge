"""Calibration and coverage checks of the Level 2 point clouds (Session 9 feasibility).

Each function returns a JSON-able record. ``scripts/pc_feasibility.py`` runs them and
``tests/simulator/test_pointcloud_alignment.py`` asserts on them.

**Diagnostics only.** These checks use simulator labels the policy never sees: segmentation,
the cube's pose and size, the table's height and exact ray casts. They measure the clouds
that :mod:`roboforge.data.pointcloud` builds from depth and calibration alone, and nothing
here feeds a cloud.

* :func:`depth_vs_ray`: rendered z-depth against an exact ``mj_ray`` cast through every pixel
  centre, over the rendered geom groups;
* :func:`known_point_check`: the cube's top-face centre, a known simulator point on a flat
  visible surface. It is projected to pixels, back-projected with the rendered depth, and
  compared with the point and its plane;
* :func:`cube_surface_residual`: the back-projected pixels that segmentation labels as the
  cube, from both cameras, against the cube's true box surface. This checks the wrist
  camera's moving extrinsics;
* :func:`table_plane_residual`: the back-projected table pixels against the table-top plane;
* :func:`cube_coverage`: how many of the cloud's sampled points lie on the cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.data.pointcloud import CloudMaker, backproject, pixel_grid
from roboforge.data.rgb import TASK_OBJECT_BODY, body_geom_ids, object_mask, project_world_point
from roboforge.data.rgb_checks import dilate as dilate_mask

# Lift's cube: one box geom (visual group 1) on the free body ``cube_main``.
CUBE_VISUAL_GEOM = "cube_g0_vis"
TABLE_VISUAL_GEOM = "table_visual"
# A sampled point counts as "on the cube" within this distance of the cube's box surface.
ON_CUBE_TOL_M = 0.002
# depth_vs_ray: largest second difference of exact inverse ray depth, relative to the pixel's
# inverse depth, for a pixel to count as locally planar (interior).
PLANAR_TOL = 0.005


def _geom(model: Any, name: str) -> int:
    import mujoco

    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if g < 0:
        raise KeyError(f"geom {name!r} not in model")
    return int(g)


def box_pose(maker: CloudMaker, state: np.ndarray, geom: str) -> dict[str, np.ndarray]:
    """World centre, rotation (columns = box axes) and half-sizes of a box geom at ``state``."""
    g = _geom(maker.model, geom)
    d = maker.depth.load(state)
    return {"pos": np.array(d.geom_xpos[g]), "rot": np.array(d.geom_xmat[g]).reshape(3, 3),
            "half": np.array(maker.model.geom_size[g])}


def box_signed_distance(points: np.ndarray, box: dict[str, np.ndarray]) -> np.ndarray:
    """Signed distance (metres) from each point to a box's surface (negative inside)."""
    local = (np.asarray(points, np.float64) - box["pos"]) @ box["rot"]
    q = np.abs(local) - box["half"]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def depth_vs_ray(maker: CloudMaker, state: np.ndarray, camera: str,
                 size: int | None = None) -> dict[str, Any]:
    """Rendered z-depth against ``mj_ray`` through every pixel centre.

    The ray direction is ``R (x, y, −1)`` in camera coordinates, so the returned distance is
    itself a z-depth. Only geom groups 1 and 2, the rendered ones, are hit.

    **Interior pixels** lie on one smooth patch of one geom, decided from geometry alone:
    * the ray hits the geom that segmentation shows there;
    * all 8 neighbours show the same geom;
    * the exact ray depth is locally planar. Inverse depth is affine in image coordinates on
      a plane, so its second differences along the row and the column through the pixel stay
      within ``PLANAR_TOL`` of the centre's inverse depth. A self-occluding mesh, such as the
      robot pedestal, has silhouettes inside one geom; this excludes them.

    The conversion is tested on interior pixels. **Boundary pixels** are the rest. There the
    multisampled depth buffer can mix two surfaces inside one pixel ("flying pixels", as in
    real depth sensors). They are reported, not tested."""
    import mujoco

    from roboforge.sim.shadow_render import ROBOSUITE_GEOMGROUP

    size = size or maker.spec.size
    dep = maker.depth.depth(state, camera, size)
    seg = maker.depth.segmentation(state, camera, size)
    cam = maker.kin.camera(state, camera, size, size)
    d = maker.depth.load(state)
    k, rot = cam["intrinsics"], np.asarray(cam["rot_world_from_cam"])
    u, v = pixel_grid(size)
    dirs = np.stack([(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"],
                     -np.ones_like(u)], -1) @ rot.T
    grp = np.array(ROBOSUITE_GEOMGROUP, dtype=np.uint8)
    gid = np.array([-1], dtype=np.int32)
    ray = np.empty(u.size)
    ray_geom = np.empty(u.size, dtype=np.int64)
    pos = np.asarray(cam["pos"], np.float64)
    for n in range(u.size):
        ray[n] = mujoco.mj_ray(maker.model, d, pos, dirs[n], grp, 1, -1, gid)
        ray_geom[n] = int(gid[0])
    geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
    label = np.where(seg[..., 1] == geom_t, seg[..., 0], -2).astype(np.int64)
    pad = np.pad(label, 1, mode="constant", constant_values=-3)
    same = np.all(np.stack([pad[a:a + size, b:b + size] for a in range(3) for b in range(3)])
                  == label[None], axis=0).reshape(-1)
    flat = dep.reshape(-1).astype(np.float64)
    hit = ray >= 0
    inv = np.where(hit, 1.0 / np.where(hit, ray, 1.0), 0.0).reshape(size, size)
    pinv = np.pad(inv, 1, mode="edge")
    d2r = np.abs(pinv[1:-1, :-2] + pinv[1:-1, 2:] - 2 * inv)
    d2c = np.abs(pinv[:-2, 1:-1] + pinv[2:, 1:-1] - 2 * inv)
    planar = (np.maximum(d2r, d2c) <= PLANAR_TOL * inv).reshape(-1)
    interior = hit & same & planar & (ray_geom == label.reshape(-1))
    err = np.abs(flat - ray)
    ei, eb = err[interior], err[hit & ~interior]
    far = maker.depth.convention["far_m"]
    return {"camera": camera, "size": size, "pixels": int(u.size), "ray_hits": int(hit.sum()),
            "ray_misses_rendered_far": int(np.sum(~hit & (flat >= 0.99 * far))),
            "ray_misses_rendered_near": int(np.sum(~hit & (flat < 0.99 * far))),
            "interior_pixels": int(interior.sum()),
            "boundary_pixels": int((hit & ~interior).sum()),
            "interior_abs_err_max_m": float(ei.max()) if ei.size else None,
            "interior_abs_err_p99_m": float(np.quantile(ei, 0.99)) if ei.size else None,
            "interior_abs_err_median_m": float(np.median(ei)) if ei.size else None,
            "interior_pixels_over_1mm": int(np.sum(ei > 1e-3)),
            "boundary_abs_err_max_m": float(eb.max()) if eb.size else None,
            "boundary_pixels_over_1mm": int(np.sum(eb > 1e-3)),
            "all_abs_err_median_m": float(np.median(err[hit])) if hit.any() else None}


def known_point_check(maker: CloudMaker, state: np.ndarray, camera: str,
                      size: int) -> dict[str, Any]:
    """Cube top-face centre → pixel → back-projected world point, against the known point.

    The known point ``P`` is the centre of the box face whose outward normal points most
    upward, from the cube geom's pose and size. ``P`` projects to continuous image
    coordinates ``(u, v)``. The pixel containing them is back-projected at its centre with its
    rendered depth, giving ``Q``. Then:

    * ``Q`` lies on the face's plane: ``plane_residual_m``, limited by depth precision;
    * ``Q`` re-projects to that pixel centre: ``reproj_err_px``, limited by floating point;
    * ``‖Q − P‖`` is at most the pixel's footprint on the plane: half the pixel diagonal,
      ``√2 / 2`` px, scaled by ``depth / f`` metres per pixel and by ``1 / cos θ``, where
      ``θ`` is the angle between the face normal and the ray. That bound is
      ``footprint_bound_m``.

    The check applies only when ``P`` is visible: its pixel shows the cube."""
    box = box_pose(maker, state, CUBE_VISUAL_GEOM)
    up = int(np.argmax(np.abs(box["rot"][2])))
    sign = np.sign(box["rot"][2, up])
    normal = sign * box["rot"][:, up]
    p_known = box["pos"] + normal * box["half"][up]
    cam = maker.kin.camera(state, camera, size, size)
    u, v, z = project_world_point(cam, p_known)
    out: dict[str, Any] = {"camera": camera, "size": size,
                           "known_point": [float(x) for x in p_known],
                           "uv": [u, v], "z_depth_m": z}
    if not (np.isfinite(u) and 0 <= u < size and 0 <= v < size):
        return {**out, "applicable": False, "reason": "point outside the image"}
    i, j = int(v), int(u)
    seg = maker_segmentation(maker, state, camera, size)
    cube_ids = body_geom_ids(maker.model, TASK_OBJECT_BODY["lift"])
    if not object_mask(seg, cube_ids)[i, j]:
        return {**out, "applicable": False, "reason": "pixel does not show the cube"}
    dep = maker.depth.depth(state, camera, size)
    q = backproject(cam, np.array([j + 0.5]), np.array([i + 0.5]), np.array([dep[i, j]]))[0]
    uq, vq, _ = project_world_point(cam, q)
    ray = q - np.asarray(cam["pos"])
    cos_t = abs(float(np.dot(ray / np.linalg.norm(ray), normal)))
    f = cam["intrinsics"]["fx"]
    bound = (np.sqrt(2) / 2) * (float(dep[i, j]) / f) / max(cos_t, 1e-6)
    return {**out, "applicable": True, "pixel": [i, j], "depth_m": float(dep[i, j]),
            "backprojected": [float(x) for x in q],
            "plane_residual_m": float(abs(np.dot(q - p_known, normal))),
            "reproj_err_px": float(np.hypot(uq - (j + 0.5), vq - (i + 0.5))),
            "dist_to_known_m": float(np.linalg.norm(q - p_known)),
            "cos_normal_ray": cos_t, "footprint_bound_m": float(bound),
            "within_footprint": bool(np.linalg.norm(q - p_known) <= bound)}


def maker_segmentation(maker: CloudMaker, state: np.ndarray, camera: str,
                       size: int) -> np.ndarray:
    """Segmentation of ``state`` (diagnostics only): ``int32 [H, W, 2]``, −1 = background.

    Rendered with the depth path's options (sites hidden), so labels match the depth pixels."""
    return maker.depth.segmentation(state, camera, size)


def _labelled_points(maker: CloudMaker, state: np.ndarray, camera: str, size: int,
                     geom_ids: np.ndarray) -> np.ndarray:
    seg = maker_segmentation(maker, state, camera, size)
    mask = object_mask(seg, geom_ids).reshape(-1)
    dep = maker.depth.depth(state, camera, size).reshape(-1)
    cam = maker.kin.camera(state, camera, size, size)
    u, v = pixel_grid(size)
    return backproject(cam, u[mask], v[mask], dep[mask])


def cube_surface_residual(maker: CloudMaker, state: np.ndarray, size: int) -> dict[str, Any]:
    """Distance of every cube-labelled back-projected pixel (each camera) to the cube's surface."""
    box = box_pose(maker, state, CUBE_VISUAL_GEOM)
    ids = np.array([_geom(maker.model, CUBE_VISUAL_GEOM)])
    out: dict[str, Any] = {"size": size}
    for cam in maker.spec.cameras:
        pts = _labelled_points(maker, state, cam, size, ids)
        sd = np.abs(box_signed_distance(pts, box)) if len(pts) else np.zeros(0)
        out[cam] = {"n": int(len(pts)),
                    "abs_max_m": float(sd.max()) if sd.size else None,
                    "abs_p99_m": float(np.quantile(sd, 0.99)) if sd.size else None,
                    "abs_median_m": float(np.median(sd)) if sd.size else None}
    return out


def table_plane_residual(maker: CloudMaker, state: np.ndarray, size: int) -> dict[str, Any]:
    """Height of every table-top pixel (each camera) against the table's top plane."""
    box = box_pose(maker, state, TABLE_VISUAL_GEOM)
    top = float(box["pos"][2] + box["half"][2])
    ids = np.array([_geom(maker.model, TABLE_VISUAL_GEOM)])
    out: dict[str, Any] = {"size": size, "table_top_z_m": top}
    for cam in maker.spec.cameras:
        pts = _labelled_points(maker, state, cam, size, ids)
        # The top face only: side faces of the slab sit below its top plane.
        on_top = pts[pts[:, 2] > top - 0.01] if len(pts) else pts
        r = np.abs(on_top[:, 2] - top) if len(on_top) else np.zeros(0)
        out[cam] = {"n": int(len(on_top)),
                    "abs_max_m": float(r.max()) if r.size else None,
                    "abs_median_m": float(np.median(r)) if r.size else None}
    return out


def cube_coverage(maker: CloudMaker, state: np.ndarray) -> dict[str, Any]:
    """The cloud at ``state`` and how many of its sampled points are on the cube."""
    cloud, info = maker.cloud(state)
    box = box_pose(maker, state, CUBE_VISUAL_GEOM)
    on = np.abs(box_signed_distance(cloud, box)) <= ON_CUBE_TOL_M
    uniq = np.unique(cloud[on], axis=0) if on.any() else np.zeros((0, 3))
    return {"n_crop": info["n_crop"], "n_in_box": info["n_in_box"],
            "on_cube_sampled": int(on.sum()), "on_cube_unique": int(len(uniq)),
            "on_cube_by_camera": {c: int(np.sum(on & (info["source_camera"] == i)))
                                  for i, c in enumerate(maker.spec.cameras)},
            "cube_z_m": float(box["pos"][2])}


def sites_in_depth(maker: CloudMaker, state: np.ndarray, camera: str,
                   size: int) -> dict[str, Any]:
    """Depth pixels that would change if site groups stayed on (why the depth path hides them)."""
    import mujoco

    from roboforge.sim.shadow_render import ROBOSUITE_GEOMGROUP

    d = maker.depth.load(state)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = ROBOSUITE_GEOMGROUP  # site groups left at MuJoCo's default
    r = mujoco.Renderer(maker.model, height=size, width=size)
    try:
        r.enable_depth_rendering()
        r.update_scene(d, camera=camera, scene_option=opt)
        r.render()  # discard the cold first render
        with_sites = np.ascontiguousarray(r.render())
    finally:
        r.close()
    hidden = maker.depth.depth(state, camera, size)
    diff = np.abs(with_sites - hidden) > 1e-6
    return {"camera": camera, "size": size, "pixels_changed_by_sites": int(diff.sum()),
            "max_change_m": float(np.abs(with_sites - hidden).max())}


__all__ = ["CUBE_VISUAL_GEOM", "ON_CUBE_TOL_M", "PLANAR_TOL", "TABLE_VISUAL_GEOM", "box_pose",
           "body_geom_ids", "box_signed_distance", "cube_coverage", "cube_surface_residual",
           "depth_vs_ray", "dilate_mask", "object_mask",
           "known_point_check", "maker_segmentation", "sites_in_depth", "table_plane_residual"]
