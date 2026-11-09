"""The detached-launch wait (D-085): transport failures are retried, everything else is not."""

from __future__ import annotations

import socket

import pytest

from roboforge.compute.launch_wait import await_result, is_poll_timeout, is_transport


def _cls(module: str, name: str) -> type[Exception]:
    return type(name, (Exception,), {"__module__": module})


ModalConnectionError = _cls("modal.exception", "ConnectionError")
StreamTerminatedError = _cls("grpclib.exceptions", "StreamTerminatedError")
ModalFunctionTimeoutError = _cls("modal.exception", "FunctionTimeoutError")


class Fake:
    """``get(timeout)`` that raises the scripted exceptions, then returns ``result``."""

    def __init__(self, script: list[BaseException], result: object) -> None:
        self.script, self.result, self.calls = list(script), result, 0

    def __call__(self, timeout: float) -> object:
        self.calls += 1
        if self.script:
            raise self.script.pop(0)
        return self.result


def test_transport_failures_and_poll_timeouts_are_ridden_out() -> None:
    sleeps: list[float] = []
    get = Fake([ModalConnectionError("[Errno 8] nodename nor servname provided"),
                ModalConnectionError("again"), TimeoutError(), StreamTerminatedError("lost"),
                socket.gaierror(8, "dns")], {"status": "complete"})
    logs: list[str] = []
    out = await_result(get, sleep=sleeps.append, clock=lambda: 0.0, log=logs.append)
    assert out == {"status": "complete"} and get.calls == 6
    assert sleeps == [20.0] * 4  # no sleep after a poll timeout
    assert sum("transport failure" in m for m in logs) == 2  # two separate outages
    assert logs[-1].endswith("after 2 transport outage(s)")


@pytest.mark.parametrize("exc", [ModalFunctionTimeoutError("remote timeout"),
                                 FileNotFoundError("nothing to resume"),
                                 ValueError("hash differs"), KeyboardInterrupt()])
def test_remote_and_local_errors_propagate_unretried(exc: BaseException) -> None:
    get = Fake([exc], None)
    with pytest.raises(type(exc)):
        await_result(get, sleep=lambda s: None, clock=lambda: 0.0, log=lambda m: None)
    assert get.calls == 1


def test_a_long_outage_gives_up_with_the_transport_error() -> None:
    t = iter(range(0, 10_000, 1000))
    get = Fake([ModalConnectionError(str(i)) for i in range(10)], None)
    with pytest.raises(ModalConnectionError):
        await_result(get, max_outage_s=2500, sleep=lambda s: None, clock=lambda: next(t),
                     log=lambda m: None)
    assert get.calls == 4  # outage starts at 0; 3000 > 2500 on the 4th failure


def test_classification() -> None:
    assert is_poll_timeout(TimeoutError()) and not is_poll_timeout(ModalFunctionTimeoutError())
    assert is_transport(ModalConnectionError()) and is_transport(StreamTerminatedError())
    assert is_transport(ConnectionResetError()) and is_transport(socket.gaierror())
    assert not is_transport(TimeoutError()) and not is_transport(OSError("disk full"))
    assert not is_transport(_cls("modal.exception", "RemoteError")())
