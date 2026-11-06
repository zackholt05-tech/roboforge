"""Construct robosuite environments from dataset metadata.

The controller configuration, control frequency and robot are taken verbatim from
the dataset's ``env_args`` so that demonstration actions are interpreted by the
exact controller that produced them. We call ``robosuite.make`` directly rather
than robomimic's ``EnvRobosuite`` wrapper because the latter imports ``egl_probe``
whenever offscreen rendering is enabled (unavailable on macOS; see DECISIONS.md).
"""

from __future__ import annotations

import copy
from typing import Any

# Render-only kwargs that must be overridden per use (not part of task semantics).
_RENDER_KWARGS = ("has_renderer", "has_offscreen_renderer", "use_camera_obs", "camera_names",
                  "camera_heights", "camera_widths", "camera_depths")


def load_sim_config(task: str) -> dict[str, Any]:
    """Load ``configs/simulator/<task>.yaml`` as an ``env_args``-shaped dict.

    The returned dict has the same structure as a dataset's ``env_args`` so it can be
    passed to :func:`make_env` without requiring the dataset to be downloaded.
    """
    import yaml

    from roboforge.paths import configs_dir

    cfg = yaml.safe_load((configs_dir() / "simulator" / f"{task}.yaml").read_text())
    return {"env_name": cfg["env_name"], "env_version": cfg["dataset_env_version"],
            "type": cfg["env_type"], "env_kwargs": cfg["env_kwargs"],
            "eval_horizon_steps": cfg["eval_horizon_steps"],
            "task_object_joints": cfg["task_object_joints"],
            "object_obs_blocks": cfg["object_obs_blocks"],
            "lagged_object_obs_blocks": cfg["lagged_object_obs_blocks"],
            "replay_success_rate_min": cfg["replay_success_rate_min"]}


def task_kwargs(env_args: dict[str, Any]) -> dict[str, Any]:
    """Return the task-defining env kwargs (dataset kwargs minus rendering settings)."""
    kw = copy.deepcopy(env_args["env_kwargs"])
    for key in _RENDER_KWARGS:
        kw.pop(key, None)
    return kw


def make_env(
    env_args: dict[str, Any],
    *,
    offscreen: bool = False,
    camera_names: list[str] | None = None,
    camera_height: int = 84,
    camera_width: int = 84,
    use_camera_obs: bool = False,
    horizon: int | None = None,
) -> Any:
    """Build a robosuite env whose task/controller settings match ``env_args`` exactly.

    Seeding (verified Session 0): in robosuite v1.5.1 the ``Lift`` and ``PickPlace``
    constructors do not accept a ``seed`` kwarg (``TypeError``), and object placement
    plus robot initialization noise draw from the *global* ``np.random`` state.
    Reproducible initial conditions therefore come from seeding ``np.random`` before
    ``reset()`` and, for evaluation, from restoring serialized MuJoCo states
    (``roboforge.sim.state_replay.reset_to_state``).
    """
    import robosuite  # imported lazily so configure_mujoco_gl() can run first

    kw = task_kwargs(env_args)
    kw.update(
        has_renderer=False,
        has_offscreen_renderer=offscreen or use_camera_obs,
        use_camera_obs=use_camera_obs,
        ignore_done=True,
    )
    if camera_names is not None:
        kw.update(camera_names=camera_names, camera_heights=camera_height,
                  camera_widths=camera_width)
    if horizon is not None:
        kw["horizon"] = horizon
    return robosuite.make(env_args["env_name"], **kw)
