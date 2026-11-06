"""Download official robomimic datasets with integrity verification.

Downloads are driven by the committed manifest ``data/manifests/robomimic_v1.5_ph.json``
(pinned Hugging Face revision + expected SHA-256). A file is accepted only if its
SHA-256 matches the manifest; a mismatching file is never silently used. Existing
files whose hash already matches are skipped, so re-running is cheap and idempotent.
Interrupted downloads resume from their ``.part`` file with an HTTP Range request.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboforge.logging_utils import get_logger
from roboforge.paths import data_root, manifests_dir
from roboforge.provenance import sha256_file

LOG = get_logger(__name__)
DEFAULT_MANIFEST = "robomimic_v1.5_ph.json"


class IntegrityError(RuntimeError):
    """Raised when a downloaded or existing file does not match its manifest hash."""


@dataclass(frozen=True)
class DatasetFile:
    """One manifest entry describing a downloadable dataset file."""

    task: str
    dataset_type: str
    hdf5_type: str
    path_in_repo: str
    local_path: str
    size_bytes: int
    sha256: str
    download_by_default: bool

    def url(self, source: dict[str, Any]) -> str:
        """Return the pinned-revision download URL for this file."""
        return source["url_template"].format(
            revision=source["hf_revision"], path_in_repo=self.path_in_repo
        )

    def destination(self, root: Path | None = None) -> Path:
        """Return the absolute local path under the data root."""
        return (root or data_root()) / self.local_path


def load_manifest(name: str = DEFAULT_MANIFEST) -> tuple[dict[str, Any], list[DatasetFile]]:
    """Load a dataset manifest and return ``(source_metadata, files)``."""
    raw = json.loads((manifests_dir() / name).read_text())
    return raw["source"], [DatasetFile(**entry) for entry in raw["files"]]


def select_files(
    files: list[DatasetFile], tasks: list[str] | None, hdf5_types: list[str] | None
) -> list[DatasetFile]:
    """Filter manifest entries; with no task filter, return the default-download set."""
    selected = [
        f
        for f in files
        if (f.task in tasks if tasks else f.download_by_default)
        and (hdf5_types is None or f.hdf5_type in hdf5_types)
    ]
    if tasks:
        unknown = set(tasks) - {f.task for f in files}
        if unknown:
            raise ValueError(f"Tasks not present in manifest: {sorted(unknown)}")
    return selected


def verify_file(entry: DatasetFile, path: Path) -> str:
    """Return the file's SHA-256, raising :class:`IntegrityError` on size or hash mismatch."""
    size = path.stat().st_size
    if size != entry.size_bytes:
        raise IntegrityError(f"{path}: size {size} != manifest {entry.size_bytes}")
    digest = sha256_file(path)
    if digest != entry.sha256:
        raise IntegrityError(f"{path}: sha256 {digest} != manifest {entry.sha256}")
    return digest


def _download_to_partial(url: str, partial: Path, expected_size: int,
                         opener: Callable[..., Any]) -> int:
    """Download ``url`` into ``partial``, resuming an interrupted ``.part`` file.

    If ``partial`` already holds ``k`` bytes (``0 < k < expected_size``), an HTTP
    ``Range: bytes=k-`` request is sent. The new bytes are appended only if the server
    answers ``206`` with a ``Content-Range`` starting at ``k``. A ``200`` answer means the
    server ignored the range, so the download restarts from zero. A ``.part`` file that is
    already larger than expected is discarded. Returns the byte offset the download resumed
    from (0 for a fresh download).
    """
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        partial.unlink()
        start = 0
    if start == expected_size:
        return start
    headers = {"Range": f"bytes={start}-"} if start else {}
    with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
        status = int(getattr(response, "status", None) or response.getcode())
        if start and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise IntegrityError(f"{url}: Content-Range {content_range!r} != offset {start}")
            mode = "ab"
        elif status == 200:
            start, mode = 0, "wb"
        else:
            raise IntegrityError(f"{url}: unexpected HTTP status {status}")
        with partial.open(mode) as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    return start


def fetch_file(
    entry: DatasetFile, source: dict[str, Any], root: Path | None = None, dry_run: bool = False,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Ensure ``entry`` exists locally with the expected hash; download it if needed.

    Downloads go to ``<dest>.part`` and resume from it after an interruption. A completed
    download is size- and SHA-256-verified before it is renamed into place. On a mismatch,
    the ``.part`` file is deleted and :class:`IntegrityError` is raised, so a corrupt file is
    never used. ``opener`` defaults to :func:`urllib.request.urlopen` (injectable for tests).
    Returns a provenance record (url, local path, sha256, action taken, resume offset).
    """
    dest = entry.destination(root)
    url = entry.url(source)
    record: dict[str, Any] = {"task": entry.task, "hdf5_type": entry.hdf5_type, "url": url,
                              "local_path": str(dest), "expected_sha256": entry.sha256}
    if dest.exists():
        try:
            record.update(sha256=verify_file(entry, dest), action="verified_existing")
            LOG.info("dataset_verified", extra={"fields": record})
            return record
        except IntegrityError as exc:
            raise IntegrityError(
                f"{exc}. Refusing to overwrite automatically; delete the file and re-run."
            ) from exc
    if dry_run:
        record.update(action="would_download", size_bytes=entry.size_bytes)
        LOG.info("dataset_dry_run", extra={"fields": record})
        return record
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    LOG.info("dataset_download_start", extra={"fields": {"url": url, "dest": str(dest)}})
    resumed_from = _download_to_partial(url, partial, entry.size_bytes,
                                        opener or urllib.request.urlopen)
    try:
        digest = verify_file(entry, partial)
    except IntegrityError:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(dest)
    record.update(sha256=digest, action="downloaded", resumed_from_bytes=resumed_from)
    LOG.info("dataset_downloaded", extra={"fields": record})
    return record
