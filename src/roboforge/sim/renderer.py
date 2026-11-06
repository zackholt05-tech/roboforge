"""Offscreen RGB rendering with an explicit, upright image convention.

robosuite's ``macros.IMAGE_CONVENTION`` defaults to ``"opengl"``, meaning raw
frames from ``sim.render`` are vertically flipped (row 0 = bottom of image).
:func:`render_rgb` always returns row 0 = top, so saved PNGs look correct.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def render_rgb(env: Any, camera_name: str, height: int, width: int) -> np.ndarray:
    """Render ``camera_name`` and return an upright ``uint8`` array of shape ``[H, W, 3]``."""
    from robosuite import macros

    frame = env.sim.render(width=width, height=height, camera_name=camera_name)
    frame = np.asarray(frame)
    if macros.IMAGE_CONVENTION == "opengl":
        frame = frame[::-1]
    if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
        raise ValueError(f"Unexpected frame {frame.dtype} {frame.shape} from camera {camera_name}")
    return np.ascontiguousarray(frame)


def camera_names(env: Any) -> list[str]:
    """Return all camera names defined in the loaded MuJoCo model."""
    model = env.sim.model
    return [model.camera_id2name(i) for i in range(model.ncam)]
