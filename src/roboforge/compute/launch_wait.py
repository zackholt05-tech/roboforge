"""Wait for a detached Modal function call, riding out client network outages (D-085).

``infra/modal_app.py`` spawns each paid launch into a *detached* ephemeral app (KI-047). Then
the remote job keeps running when the client loses its network, and the client polls the
call's result until it arrives. This module holds the polling rule. It needs no Modal import,
so the venv's tests can exercise it.

Only **transport** failures are retried:
* Modal's ``ConnectionError`` (DNS loss arrives as one);
* grpclib's ``StreamTerminatedError``;
* socket ``gaierror`` and the builtin connection errors.

The poll's own timeout (the builtin ``TimeoutError`` that ``FunctionCall.get(timeout=…)``
raises when no output is ready) just means "not finished yet". Anything else, including every
exception the remote job raises, propagates unchanged. ``error``, ``interrupted`` and
``complete`` keep their meanings. A continuous outage longer than ``max_outage_s`` re-raises
the last transport error. The detached job keeps running, and the call can be re-attached
later by its ID.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from typing import Any

TRANSPORT_TYPES = {("modal.exception", "ConnectionError"),
                   ("grpclib.exceptions", "StreamTerminatedError")}
_OS_TRANSPORT = (socket.gaierror, ConnectionResetError, ConnectionAbortedError,
                 ConnectionRefusedError, BrokenPipeError)


def is_poll_timeout(exc: BaseException) -> bool:
    """``FunctionCall.get(timeout=…)`` found no output yet (the builtin ``TimeoutError``)."""
    return type(exc) is TimeoutError


def is_transport(exc: BaseException) -> bool:
    """A client-side network failure, never a result of the remote job."""
    t = type(exc)
    return (t.__module__, t.__name__) in TRANSPORT_TYPES or isinstance(exc, _OS_TRANSPORT)


def await_result(get: Callable[[float], Any], *, poll_s: float = 30.0,
                 retry_sleep_s: float = 20.0, max_outage_s: float = 3600.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] = print) -> Any:
    """Poll ``get(timeout)`` until it returns; retry transport failures (see module doc)."""
    outage_start: float | None = None
    outages = 0
    while True:
        try:
            out = get(poll_s)
        except BaseException as exc:
            if is_poll_timeout(exc):
                outage_start = None
                continue
            if not is_transport(exc):
                raise
            now = clock()
            if outage_start is None:
                outage_start, outages = now, outages + 1
                log(f"launch_wait: transport failure #{outages} "
                    f"({type(exc).__name__}: {exc}); retrying")
            if now - outage_start > max_outage_s:
                raise
            sleep(retry_sleep_s)
            continue
        if outages:
            log(f"launch_wait: result received after {outages} transport outage(s)")
        return out


__all__ = ["TRANSPORT_TYPES", "await_result", "is_poll_timeout", "is_transport"]
