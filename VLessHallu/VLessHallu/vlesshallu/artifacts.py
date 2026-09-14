from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class JsonlLedger:
    def __init__(self, path: str | Path, *, id_key: str):
        self.path = Path(path)
        self.id_key = id_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = str(record[self.id_key])
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(
                        f"invalid ledger record at {self.path}:{line_number}: {error}"
                    ) from error
                records[key] = record
        return records

    def contains(self, sample_id: str | int) -> bool:
        return str(sample_id) in self._records

    def get(self, sample_id: str | int) -> dict[str, Any] | None:
        return self._records.get(str(sample_id))

    def append(self, record: Mapping[str, Any]) -> None:
        key = str(record[self.id_key])
        if key in self._records:
            raise ValueError(f"duplicate {self.id_key}={key} in {self.path}")
        serialized = json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = dict(record)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())


class RunLock:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / ".active.lock"
        self._acquired = False

    def __enter__(self) -> "RunLock":
        payload = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time()}
        )
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise RuntimeError(f"run is already active: {self.path.parent}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False


@dataclass
class RunStore:
    path: Path
    captions: JsonlLedger
    pruning: JsonlLedger

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        run_hash: str,
        manifest: Mapping[str, Any],
    ) -> "RunStore":
        path = Path(run_root) / run_hash
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            existing = read_json(manifest_path)
            if existing != dict(manifest):
                raise RuntimeError(f"run hash collision or changed manifest at {path}")
        else:
            atomic_write_json(manifest_path, dict(manifest))
        return cls(
            path=path,
            captions=JsonlLedger(path / "captions.jsonl", id_key="sample_id"),
            pruning=JsonlLedger(path / "pruning.jsonl", id_key="sample_id"),
        )

    def pending(self, samples: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [sample for sample in samples if not self.captions.contains(sample["sample_id"])]

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        atomic_write_json(self.path / "metrics.json", dict(metrics))

    def write_runtime(self, runtime: Mapping[str, Any]) -> None:
        atomic_write_json(self.path / "runtime.json", dict(runtime))
