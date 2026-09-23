"""Object-storage abstraction. ``LocalStorage`` for development; the interface mirrors S3 put/get/delete
so an S3-compatible backend can be added without touching callers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Protocol


class StorageError(RuntimeError):
    pass


class UploadTooLarge(StorageError):
    pass


class Storage(Protocol):
    backend: str

    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[int, str]: ...
    def path_for(self, key: str) -> Path: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def usage_bytes(self) -> int: ...


class LocalStorage:
    backend = "local"

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise StorageError("storage key escapes the storage root")
        return p

    def path_for(self, key: str) -> Path:
        return self._resolve(key)

    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> tuple[int, str]:
        """Stream to a temp file, hashing as we go; enforce the size limit; atomic rename."""
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        h = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as fh:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLarge(f"upload exceeds {max_bytes // (1024 * 1024)} MB")
                    h.update(chunk)
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        return size, h.hexdigest()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> bool:
        """Overwrite then unlink (best-effort secure deletion on a local filesystem)."""
        p = self._resolve(key)
        if not p.is_file():
            return False
        size = p.stat().st_size
        with p.open("r+b") as fh:
            remaining = size
            block = b"\0" * (1024 * 1024)
            while remaining > 0:
                n = min(remaining, len(block))
                fh.write(block[:n])
                remaining -= n
            fh.flush()
            os.fsync(fh.fileno())
        p.unlink()
        return True

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
