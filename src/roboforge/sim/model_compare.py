"""Diagnostics for KNOWN_ISSUES KI-014 / KI-016: compare compiled models and mocap markers.

Used only by checks (``scripts/check_sim_contract.py`` and tests), never on the evaluation
path. Evaluation always compiles the stored XML string as is (DECISIONS D-025).

* :func:`strip_markers` / :func:`insert_markers` remove or add robosuite 1.5.1's two
  non-colliding mocap bodies (``left_eef_target``, ``right_eef_target``). They work with plain
  text substitution on MuJoCo-saved XML and do not parse it.
* :func:`param_diff` compares two compiled ``MjModel`` objects field by field. Bodies, geoms,
  joints, actuators and sites are aligned **by name**, because the markers shift body and geom
  IDs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

MARKER_BODIES = ("left_eef_target", "right_eef_target")
_MARKER = re.compile(r'\s*<body name="(?:left|right)_eef_target"[^>]*>.*?</body>', re.S)


def marker_snippets(xml: str) -> list[str]:
    """The two marker ``<body>`` elements of a MuJoCo-saved robosuite model."""
    found = [m.group(0) for m in _MARKER.finditer(xml)]
    if len(found) != 2:
        raise ValueError(f"expected 2 marker bodies, found {len(found)}")
    return found


def strip_markers(xml: str) -> str:
    """``xml`` without the two marker bodies."""
    out, n = _MARKER.subn("", xml)
    if n != 2:
        raise ValueError(f"expected to remove 2 marker bodies, removed {n}")
    return out


def insert_markers(xml: str, snippets: list[str]) -> str:
    """``xml`` with the marker bodies added at the start of ``<worldbody>``."""
    if xml.count("<worldbody>") != 1 or any(f'name="{b}"' in xml for b in MARKER_BODIES):
        raise ValueError("model must have one <worldbody> and no markers yet")
    return xml.replace("<worldbody>", "<worldbody>" + "".join(snippets), 1)


_GROUPS = {"body_": ("mjOBJ_BODY", "nbody"), "geom_": ("mjOBJ_GEOM", "ngeom"),
           "jnt_": ("mjOBJ_JOINT", "njnt"), "actuator_": ("mjOBJ_ACTUATOR", "nu"),
           "site_": ("mjOBJ_SITE", "nsite"), "dof_": (None, "nv")}
# ID-valued fields whose values legitimately shift when bodies/geoms are inserted.
INDEX_FIELDS = {"body_parentid", "body_rootid", "body_weldid", "body_geomadr", "body_jntadr",
                "body_dofadr", "body_treeid", "body_mocapid", "geom_bodyid", "jnt_bodyid",
                "site_bodyid", "dof_bodyid", "dof_treeid", "actuator_trnid", "body_conaffinity",
                "body_contype", "body_bvhadr", "geom_matid", "site_matid", "body_geomnum",
                "body_bvhnum", "geom_dataid", "dof_parentid", "jnt_qposadr", "jnt_dofadr",
                "dof_jntid", "body_margin"}


def _excluded(model: Any, bodies: tuple[str, ...]) -> dict[str, set[Any]]:
    """Names (geoms, joints) and DOF indices belonging to ``bodies`` in ``model``."""
    import mujoco

    o = mujoco.mjtObj
    ids = {mujoco.mj_name2id(model, o.mjOBJ_BODY, b) for b in bodies} - {-1}
    return {"mjOBJ_BODY": set(bodies),
            "mjOBJ_GEOM": {mujoco.mj_id2name(model, o.mjOBJ_GEOM, g) for g in range(model.ngeom)
                           if int(model.geom_bodyid[g]) in ids},
            "mjOBJ_JOINT": {mujoco.mj_id2name(model, o.mjOBJ_JOINT, j)
                            for j in range(model.njnt) if int(model.jnt_bodyid[j]) in ids},
            "mjOBJ_SITE": {mujoco.mj_id2name(model, o.mjOBJ_SITE, s) for s in range(model.nsite)
                           if int(model.site_bodyid[s]) in ids},
            "dof": {d for d in range(model.nv) if int(model.dof_bodyid[d]) in ids}}


def param_diff(a: Any, b: Any, exclude_bodies: tuple[str, ...] = ()) -> dict[str, Any]:
    """Name-aligned max |difference| of every numeric per-element field of two models.

    ``exclude_bodies`` drops those bodies and their geoms, joints, sites and DOFs from the
    comparison. Example: Lift's cube, whose size robosuite samples at every reset.
    """
    import mujoco

    out: dict[str, Any] = {"only_in_a": [], "only_in_b": [], "fields": {},
                           "excluded_bodies": list(exclude_bodies)}
    ex = _excluded(a, exclude_bodies)
    for prefix, (obj, count) in _GROUPS.items():
        na, nb = getattr(a, count), getattr(b, count)
        if obj is None:
            if na != nb:
                out["fields"][f"{prefix}*"] = f"count {na} != {nb}"
                continue
            ia = ib = [i for i in range(na) if i not in ex["dof"]]
            names = [str(i) for i in ia]
        else:
            o = getattr(mujoco.mjtObj, obj)
            la = [mujoco.mj_id2name(a, o, i) for i in range(na)]
            lb = [mujoco.mj_id2name(b, o, i) for i in range(nb)]
            names = [n for n in la if n is not None and n in lb
                     and n not in ex.get(obj, set())]
            out["only_in_a"] += sorted(n for n in set(la) - set(lb) if n)
            out["only_in_b"] += sorted(n for n in set(lb) - set(la) if n)
            ia, ib = [la.index(n) for n in names], [lb.index(n) for n in names]
        for f in sorted(dir(a)):
            if not f.startswith(prefix):
                continue
            va, vb = getattr(a, f, None), getattr(b, f, None)
            if (not isinstance(va, np.ndarray) or va.ndim == 0 or va.shape[0] != na
                    or vb.shape[0] != nb or va.dtype.kind not in "fiub"):
                continue
            d = np.abs(va[ia].astype(np.float64) - vb[ib].astype(np.float64))
            if d.size and float(d.max()) > 0:
                row = int(np.unravel_index(int(d.argmax()), d.shape)[0])
                out["fields"][f] = {"max_abs": float(d.max()), "at": names[row],
                                    "index_field": f in INDEX_FIELDS}
    out["opt_equal"] = all(np.array_equal(np.asarray(getattr(a.opt, k)),
                                          np.asarray(getattr(b.opt, k)))
                           for k in ("timestep", "impratio", "cone", "solver", "iterations",
                                     "tolerance", "gravity", "integrator"))
    return out
