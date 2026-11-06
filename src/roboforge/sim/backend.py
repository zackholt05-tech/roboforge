"""Select the MuJoCo OpenGL backend *before* mujoco/robosuite are imported.

MuJoCo reads ``MUJOCO_GL`` at import time, and robosuite v1.5.1
(``robosuite/utils/binding_utils.py``) overwrites it with ``cgl`` on macOS and
``egl`` on Linux unless it is already ``osmesa`` or ``glx``. Supported choices:

* macOS (Session 0 verified): ``cgl`` — native headless CGL context, no display needed.
* Linux with an NVIDIA GPU + EGL driver libraries: ``egl``.
* Linux CPU-only container: ``osmesa`` (software rasterizer, requires libOSMesa).

Call :func:`configure_mujoco_gl` at the very top of any entrypoint that renders.
"""

from __future__ import annotations

import os
import sys

VALID_BACKENDS = {"darwin": ("cgl", "glfw"), "linux": ("egl", "osmesa", "glx")}


def default_backend(platform: str = sys.platform) -> str:
    """Return the default headless backend for ``platform``."""
    if platform == "darwin":
        return "cgl"
    if platform.startswith("linux"):
        return "egl"
    raise RuntimeError(f"Unsupported platform for headless MuJoCo rendering: {platform}")


def configure_mujoco_gl(requested: str | None = None) -> str:
    """Set ``MUJOCO_GL`` (explicit argument > existing env var > platform default).

    Raises:
        RuntimeError: if mujoco was already imported with a different backend, or
            if the backend is not valid for this platform.
    """
    backend = (requested or os.environ.get("MUJOCO_GL") or default_backend()).lower()
    platform_key = "darwin" if sys.platform == "darwin" else "linux"
    if backend not in VALID_BACKENDS[platform_key]:
        raise RuntimeError(
            f"MUJOCO_GL={backend!r} is not valid on {sys.platform}; "
            f"choose one of {VALID_BACKENDS[platform_key]}"
        )
    if "mujoco" in sys.modules and os.environ.get("MUJOCO_GL", "").lower() != backend:
        raise RuntimeError(
            "mujoco was imported before configure_mujoco_gl(); the backend cannot change now."
        )
    os.environ["MUJOCO_GL"] = backend
    if backend == "osmesa":
        # PyOpenGL must also use the OSMesa platform for MuJoCo's osmesa context.
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    elif backend == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return backend
