"""Bind the frozen training and evaluation code to the X2 (Can MH) data (DECISIONS D-121).

The frozen code finds a task's data through ``roboforge.data.pipeline.load_task_artifacts``
(the ``*_ph`` paths) and builds training windows in ``roboforge.train.run.load_windows`` (the
PH processed file). :func:`x2_bindings` rebinds exactly these module attributes, only inside
the ``with`` block, to the integrity-checked X2 versions:

* ``load_task_artifacts`` in :mod:`roboforge.data.pipeline` (so the frozen ``load_specs`` and
  ``SequenceDataset.from_committed`` follow), :mod:`roboforge.train.run`,
  :mod:`roboforge.eval.learned` and :mod:`roboforge.sim.env_adapter`;
* :func:`roboforge.train.run.load_windows`.

No frozen file is edited. Inside the context the frozen ``train_run`` records the X2 data
hashes and ``dataset_id``, ``RoboForgeEnv`` carries the X2 task spec (horizon 500, D-120), and
``LearnedPolicy`` checks a checkpoint against the X2 normalization and specs. On exit, also on
an error, the frozen functions are restored. Only X2 entry scripts enter the context.

``loader`` (tests only) replaces the X2 loader, for example with the Can PH loader to show that
the context changes nothing but the data source.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

# (module, attribute) pairs rebound to the X2 artifacts loader.
LOADER_SITES = (("roboforge.data.pipeline", "load_task_artifacts"),
                ("roboforge.train.run", "load_task_artifacts"),
                ("roboforge.eval.learned", "load_task_artifacts"),
                ("roboforge.sim.env_adapter", "load_task_artifacts"))
WINDOWS_SITE = ("roboforge.train.run", "load_windows")
_ACTIVE: list[str] = []


def active() -> bool:
    """True inside :func:`x2_bindings`."""
    return bool(_ACTIVE)


def x2_windows_factory(loader: Callable[..., dict[str, Any]],
                       episodes_of: Callable[[dict[str, Any]], dict[str, dict[str, np.ndarray]]]
                       ) -> Callable[[Any], tuple[Any, Any]]:
    """A ``load_windows(cfg)`` that reads the artifacts from ``loader`` and the episodes from
    ``episodes_of(docs)``, with the frozen ``SequenceDataset`` and ``WindowTensors``.

    Refuses a split with a flagged episode, as ``SequenceDataset.from_committed`` does, unless
    the split accepts that flag with exactly its reason (D-122)."""
    from roboforge.data.normalization import Normalizer
    from roboforge.data.sequence import SequenceDataset
    from roboforge.train.data import WindowTensors

    def load_windows(cfg: Any) -> tuple[Any, Any]:
        docs = loader(cfg.task)
        sp = docs["split"]
        episodes = episodes_of(docs)
        out = []
        for split in ("train", "valid"):
            demos = sp["splits"][split]
            acc = sp.get("accepted_flags", {})
            flagged = sorted(d for d in set(demos) & set(sp["flagged"])
                             if acc.get(d, {}).get("reason") != sp["flagged"][d])
            if flagged:
                raise ValueError(f"X2 {split}: flagged episodes {flagged}")
            ds = SequenceDataset(episodes, demos,
                                 Normalizer.from_dict(docs["normalization"]["obs"]),
                                 Normalizer.from_dict(docs["normalization"]["action"]),
                                 cfg.horizons.obs, cfg.horizons.pred)
            out.append(WindowTensors.from_dataset(ds))
        return out[0], out[1]

    return load_windows


@contextmanager
def x2_bindings(loader: Callable[..., dict[str, Any]] | None = None,
                episodes_of: Callable[[dict[str, Any]], dict[str, dict[str, np.ndarray]]]
                | None = None) -> Iterator[None]:
    """Rebind the frozen data lookups to the X2 artifacts for the duration of the block."""
    from roboforge.x2 import data as x2data

    if _ACTIVE:
        raise RuntimeError("x2_bindings is not re-entrant")
    loader = loader or x2data.load_x2_artifacts
    episodes_of = episodes_of or x2data.load_x2_episodes
    windows = x2_windows_factory(loader, episodes_of)
    saved: list[tuple[Any, str, Any]] = []
    try:
        for mod_name, attr in LOADER_SITES:
            mod = importlib.import_module(mod_name)
            saved.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, loader)
        mod = importlib.import_module(WINDOWS_SITE[0])
        saved.append((mod, WINDOWS_SITE[1], getattr(mod, WINDOWS_SITE[1])))
        setattr(mod, WINDOWS_SITE[1], windows)
        _ACTIVE.append("x2")
        yield
    finally:
        for mod, attr, original in reversed(saved):
            setattr(mod, attr, original)
        _ACTIVE.clear()
