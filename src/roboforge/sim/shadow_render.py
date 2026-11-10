"""Offscreen rendering of any flattened state without touching the live simulation.

``ShadowRenderer`` shares the compiled ``MjModel`` with the environment but draws from its
own scratch ``MjData``: the state is copied in and ``mj_forward`` runs **on the scratch data
only**. Rendering through robosuite's context instead would show kinematics one physics
substep stale after ``env.step`` (KI-015), and refreshing the live data with ``mj_forward``
changes the solver warm start and therefore the rest of the rollout (DECISIONS D-024).

The visual options reproduce robosuite's offscreen context (``render_collision_mesh=False``,
``render_visual_mesh=True``: geom group 0 hidden, groups 1–2 shown). Session 2 probe: frames
equal robosuite's renders exactly on Lift (agentview, frontview, eye-in-hand) and within one
grey level on 27 of 65,536 agentview pixels on Can. Images are upright (row 0 = top), the
same convention as ``roboforge.sim.renderer.render_rgb``. Model-level visual edits made by
robosuite (site colours) are seen because the model object is shared.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# robosuite MjRenderContext: geomgroup[0] = render_collision_mesh, [1] = render_visual_mesh.
ROBOSUITE_GEOMGROUP = (0, 1, 1, 0, 0, 0)


class ShadowRenderer:
    """RGB and segmentation renders of flattened states for one compiled model."""

    def __init__(self, model: Any) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        self._data = mujoco.MjData(model)
        self._renderers: dict[tuple[int, int], Any] = {}
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = ROBOSUITE_GEOMGROUP

    def _renderer(self, height: int, width: int) -> Any:
        key = (int(height), int(width))
        if key not in self._renderers:
            self._renderers[key] = self._mj.Renderer(self.model, height=key[0], width=key[1])
        return self._renderers[key]

    def _load(self, state: np.ndarray) -> None:
        m, d = self.model, self._data
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (1 + m.nq + m.nv,):
            raise ValueError(f"state shape {state.shape} != ({1 + m.nq + m.nv},)")
        d.time = state[0]
        d.qpos[:] = state[1:1 + m.nq]
        d.qvel[:] = state[1 + m.nq:]
        self._mj.mj_forward(m, d)

    def rgb(self, state: np.ndarray, camera: str, height: int, width: int) -> np.ndarray:
        """Upright ``uint8 [H, W, 3]`` image of ``state`` from the named camera."""
        self._load(state)
        r = self._renderer(height, width)
        r.update_scene(self._data, camera=camera, scene_option=self._opt)
        frame = np.ascontiguousarray(r.render())
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError(f"unexpected frame {frame.dtype} {frame.shape} from {camera}")
        return frame

    def segmentation(self, state: np.ndarray, camera: str, height: int,
                     width: int, sites: bool = True) -> np.ndarray:
        """``int32 [H, W, 2]``: per pixel (object id, ``mjtObj`` type); ``-1`` = background.

        ``sites=False`` hides every site group. Segmentation draws sites whatever their alpha,
        so robosuite's transparent gripper sites (``grip_site_cylinder``: radius 5 mm,
        half-length 10 m) would occlude geoms that the RGB render shows (DECISIONS D-091)."""
        self._load(state)
        r = self._renderer(height, width)
        opt = self._opt
        if not sites:
            opt = self._mj.MjvOption()
            opt.geomgroup[:] = ROBOSUITE_GEOMGROUP
            opt.sitegroup[:] = 0
        r.enable_segmentation_rendering()
        try:
            r.update_scene(self._data, camera=camera, scene_option=opt)
            seg = np.ascontiguousarray(r.render())
        finally:
            r.disable_segmentation_rendering()
        return seg

    def close(self) -> None:
        """Release the GL contexts."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
