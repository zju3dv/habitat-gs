"""JSONL logging helpers for minimal RPF episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append dict records as JSON lines."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
