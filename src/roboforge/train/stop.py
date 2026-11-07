"""Cooperative stop requests for a training segment (Session 5).

A stop is never acted on in the middle of an optimizer step. Signal handlers and the deadline
only set a flag. :meth:`roboforge.train.loop.Trainer.fit` checks it after each completed step,
writes a resume checkpoint, and returns. This keeps the saved state consistent: an exception
raised inside ``backward()`` or ``optimizer.step()`` could leave half-updated parameters.

Sources of a stop request:

* ``SIGINT``/``SIGTERM`` (a local Ctrl-C, or a Modal preemption, which sends SIGINT and
  SIGKILL 30 s later), through :meth:`StopControl.handle_signals`;
* a wall-clock deadline, set below a remote job's hard timeout so that the job stops itself
  cleanly instead of being killed;
* ``stop_at_step``, used only by tests and the local remote-layout simulation to interrupt a
  run at an exact step.

Signal handlers can only be installed from the main thread. Elsewhere,
:meth:`StopControl.handle_signals` installs nothing and records that, and an interrupt then
arrives as ``KeyboardInterrupt``. The run then resumes from its last indexed checkpoint,
because the in-memory state may be inconsistent and is not saved.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Iterator


class StopControl:
    """A flag set by signals, a deadline or an exact step."""

    def __init__(self, deadline_s: float | None = None,
                 stop_at_step: int | None = None) -> None:
        self.reason: str | None = None
        self.deadline_s = deadline_s
        self._deadline = (time.monotonic() + deadline_s) if deadline_s is not None else None
        self.stop_at_step = stop_at_step
        self.signals_installed = False

    def request(self, reason: str) -> None:
        """Ask the trainer to stop after the current step (the first reason wins)."""
        if self.reason is None:
            self.reason = reason

    def check(self, step: int) -> str | None:
        """Called after every completed step; returns the stop reason, if any."""
        if self.stop_at_step is not None and step >= self.stop_at_step:
            self.request(f"stop_at_step {self.stop_at_step}")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self.request(f"deadline {self.deadline_s:.0f} s")
        return self.reason

    @contextlib.contextmanager
    def handle_signals(self, signums: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)
                       ) -> Iterator[StopControl]:
        """Turn the given signals into stop requests while the block runs."""
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        old = {s: signal.getsignal(s) for s in signums}

        def handler(signum: int, _frame: object) -> None:
            self.request(f"signal {signal.Signals(signum).name}")

        for s in signums:
            signal.signal(s, handler)
        self.signals_installed = True
        try:
            yield self
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            self.signals_installed = False


__all__ = ["StopControl"]
