from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlEventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = Lock()

    def write(self, event_type: str, **payload: Any) -> None:
        record = {"event": event_type, "time_unix": time.time(), **payload}
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            self._handle.flush()

    def byte_offset(self) -> int:
        with self._lock:
            self._handle.flush()
            return self.path.stat().st_size

    def truncate_to(self, byte_offset: int) -> None:
        """Discard events written after the controller checkpoint being resumed."""
        with self._lock:
            self._handle.flush()
            current_size = self.path.stat().st_size
            if byte_offset < 0 or byte_offset > current_size:
                raise ValueError(
                    f"invalid event-log checkpoint offset {byte_offset}; size is {current_size}"
                )
            self._handle.close()
            with self.path.open("r+b") as binary_handle:
                binary_handle.truncate(byte_offset)
            self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            self._handle.close()
