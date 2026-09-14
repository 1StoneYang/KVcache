from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


_CHUNK_SIZE = 1024 * 1024
_MAX_STALLED_ATTEMPTS = 12


def download_zip(url: str, destination: Path) -> Path:
    return _download(
        url,
        destination,
        validator=lambda path, size: _valid_zip(path, size),
    )


def download_file(
    url: str, destination: Path, *, expected_sha256: str | None = None
) -> Path:
    return _download(
        url,
        destination,
        validator=lambda path, size: _valid_file(
            path, size, expected_sha256=expected_sha256
        ),
    )


def _download(
    url: str,
    destination: Path,
    *,
    validator: Callable[[Path, int], bool],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _content_length(url)
    if destination.is_file() and validator(destination, expected):
        return destination

    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.is_file():
        if partial.is_file() and partial.stat().st_size > destination.stat().st_size:
            destination.unlink()
        else:
            os.replace(destination, partial)

    stalled_attempts = 0
    while True:
        current = partial.stat().st_size if partial.is_file() else 0
        if current == expected:
            break
        if current > expected:
            raise RuntimeError(
                f"partial download is larger than the remote file: {partial}"
            )

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "VLessHallu/0.2",
                "Range": f"bytes={current}-",
            },
        )
        before = current
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if current and status != 206:
                    current = 0
                    mode = "wb"
                else:
                    _validate_content_range(response.headers.get("Content-Range"), current)
                    mode = "ab" if current else "wb"
                with partial.open(mode) as output:
                    while True:
                        try:
                            chunk = response.read(_CHUNK_SIZE)
                        except http.client.IncompleteRead as error:
                            chunk = error.partial
                        if chunk:
                            output.write(chunk)
                        if len(chunk) < _CHUNK_SIZE:
                            break
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            pass

        after = partial.stat().st_size if partial.is_file() else 0
        if after > before:
            stalled_attempts = 0
            print(f"downloaded {destination.name}: {after}/{expected} bytes", flush=True)
        else:
            stalled_attempts += 1
            if stalled_attempts >= _MAX_STALLED_ATTEMPTS:
                raise RuntimeError(
                    f"download stalled after {_MAX_STALLED_ATTEMPTS} attempts: {url}"
                )
            time.sleep(min(2**stalled_attempts, 30))

    if not validator(partial, expected):
        raise RuntimeError(f"downloaded file failed validation: {partial}")
    os.replace(partial, destination)
    return destination


def _content_length(url: str) -> int:
    request = urllib.request.Request(
        url, headers={"User-Agent": "VLessHallu/0.2"}, method="HEAD"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        value = response.headers.get("Content-Length")
    if value is None or int(value) <= 0:
        raise RuntimeError(f"remote server did not provide Content-Length: {url}")
    return int(value)


def _validate_content_range(value: str | None, expected_start: int) -> None:
    if expected_start == 0:
        return
    if value is None or not value.startswith(f"bytes {expected_start}-"):
        raise RuntimeError(
            f"server returned an invalid Content-Range for offset {expected_start}: {value}"
        )


def _valid_zip(path: Path, expected_size: int) -> bool:
    if not _valid_file(path, expected_size):
        return False
    try:
        with zipfile.ZipFile(path) as bundle:
            return bundle.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _valid_file(
    path: Path, expected_size: int, *, expected_sha256: str | None = None
) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if expected_sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * _CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256
