"""Resumable, hash-verified downloads with a fake HTTP opener (no network)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from roboforge.data.fetch import DatasetFile, IntegrityError, fetch_file

PAYLOAD = bytes(range(256)) * 40
SOURCE = {"url_template": "https://example.invalid/{revision}/{path_in_repo}", "hf_revision": "r"}


class FakeResponse(io.BytesIO):
    """Minimal stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, body: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(body)
        self.status, self.headers = status, headers


class FakeServer:
    """Serves PAYLOAD; honours Range requests unless ``supports_range`` is False."""

    def __init__(self, supports_range: bool = True, body: bytes = PAYLOAD) -> None:
        self.supports_range, self.body, self.requests = supports_range, body, []

    def __call__(self, request, timeout: float) -> FakeResponse:
        rng = request.headers.get("Range")
        self.requests.append(rng)
        if rng and self.supports_range:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return FakeResponse(self.body[start:], 206,
                                {"Content-Range": f"bytes {start}-{len(self.body) - 1}/*"})
        return FakeResponse(self.body, 200, {})


def _entry(sha: str | None = None) -> DatasetFile:
    return DatasetFile(task="t", dataset_type="ph", hdf5_type="low_dim", path_in_repo="x.hdf5",
                       local_path="raw/x.hdf5", size_bytes=len(PAYLOAD),
                       sha256=sha or hashlib.sha256(PAYLOAD).hexdigest(),
                       download_by_default=True)


def _partial(root: Path, n: int) -> Path:
    p = root / "raw" / "x.hdf5.part"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(PAYLOAD[:n])
    return p


def test_fresh_download_is_verified_and_renamed(tmp_path: Path) -> None:
    server = FakeServer()
    rec = fetch_file(_entry(), SOURCE, root=tmp_path, opener=server)
    assert rec["action"] == "downloaded" and rec["resumed_from_bytes"] == 0
    assert (tmp_path / "raw" / "x.hdf5").read_bytes() == PAYLOAD
    assert server.requests == [None]


def test_interrupted_download_resumes_with_range(tmp_path: Path) -> None:
    _partial(tmp_path, 3000)
    server = FakeServer()
    rec = fetch_file(_entry(), SOURCE, root=tmp_path, opener=server)
    assert rec["resumed_from_bytes"] == 3000 and server.requests == ["bytes=3000-"]
    assert (tmp_path / "raw" / "x.hdf5").read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts_from_zero(tmp_path: Path) -> None:
    _partial(tmp_path, 3000)
    rec = fetch_file(_entry(), SOURCE, root=tmp_path, opener=FakeServer(supports_range=False))
    assert rec["resumed_from_bytes"] == 0
    assert (tmp_path / "raw" / "x.hdf5").read_bytes() == PAYLOAD


def test_complete_partial_is_verified_without_a_request(tmp_path: Path) -> None:
    _partial(tmp_path, len(PAYLOAD))
    server = FakeServer()
    rec = fetch_file(_entry(), SOURCE, root=tmp_path, opener=server)
    assert rec["action"] == "downloaded" and server.requests == []


def test_hash_mismatch_deletes_partial_and_raises(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError):
        fetch_file(_entry(sha="0" * 64), SOURCE, root=tmp_path, opener=FakeServer())
    assert not (tmp_path / "raw" / "x.hdf5").exists()
    assert not (tmp_path / "raw" / "x.hdf5.part").exists()


def test_corrupt_resumed_bytes_are_caught_by_the_hash(tmp_path: Path) -> None:
    p = _partial(tmp_path, 3000)
    p.write_bytes(b"\xff" * 3000)  # wrong prefix on disk
    with pytest.raises(IntegrityError):
        fetch_file(_entry(), SOURCE, root=tmp_path, opener=FakeServer())
    assert not p.exists()


def test_existing_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "x.hdf5"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)
    server = FakeServer()
    assert fetch_file(_entry(), SOURCE, root=tmp_path, opener=server)["action"] == \
        "verified_existing"
    assert server.requests == []
