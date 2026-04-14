"""SpatialMemory — wraps the existing spatial_memory_*.json behind
the Memory protocol.

The bridge creates and updates the spatial_memory JSON file via
`_apply_spatial_memory_append` (in `mixins_patch.py`). This class
is **read-only from the agent side** — it only reads the file for
prompt summarization and querying. The `add()` method is a no-op
because the agent submits spatial_memory_append entries through the
`update_nav_status` tool, which the bridge validates and persists.

The `summarize()` output is designed to match the existing
`PromptBuilder._format_spatial_summary` exactly, so the transition
from inline I/O to MemoryBundle is behaviourally invisible to the
LLM.

File format (managed by bridge, not changed by Phase 3)::

    {
        "task_id": "navloop-abc",
        "grid_resolution_m": 0.5,
        "snapshots": [
            {"heading_deg": 90, "scene_description": "...",
             "room_label": "kitchen", "objects_detected": ["table"],
             "ts": "2026-04-10T...Z"},
            ...
        ],
        "rooms": {
            "kitchen": {"visit_count": 2, "first_seen": "...", "last_seen": "..."},
            ...
        },
        "object_sightings": {
            "table": {"count": 1, "last_seen": "..."},
            ...
        }
    }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _empty_spatial() -> Dict[str, Any]:
    """Return a fresh empty spatial memory structure every time.

    Previously this was a module-level dict constant with
    ``dict(_EMPTY_SPATIAL)`` shallow-copies in ``_load()``. That
    shared the nested ``[]`` / ``{}`` objects across calls — if any
    caller mutated the returned structure (e.g. via ``export()``),
    subsequent fallback loads would contain phantom data. Returning
    a fresh literal on every call eliminates the shared-state risk.
    """
    return {"snapshots": [], "rooms": {}, "object_sightings": {}}


class SpatialMemory:
    """Read-only wrapper for spatial_memory_*.json.

    Satisfies the ``Memory`` protocol defined in ``base.py``.
    """

    name: str = "spatial"
    persistence: str = "per-loop"

    def __init__(self, spatial_memory_file: str):
        self.file = spatial_memory_file

    # ── I/O ──────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        """Read the JSON file. Returns empty structure on any failure
        (missing file, permission error, malformed JSON). Non-fatal
        because the prompt can still render a "no memory" placeholder
        — losing a snapshot is better than crashing the agent."""
        if not self.file or not os.path.isfile(self.file):
            return _empty_spatial()
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return _empty_spatial()
            return data
        except Exception:
            return _empty_spatial()

    # ── Memory protocol ──────────────────────────────────────────

    def add(self, entry: Dict[str, Any]) -> None:
        """No-op. The bridge handles all writes via
        ``_apply_spatial_memory_append``. The agent sends entries
        through ``update_nav_status(spatial_memory_append=[...])``
        and the bridge validates + persists."""
        pass

    def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve snapshots, optionally filtered by room or object.

        Supported filter keys:
          - ``room``: match snapshots whose ``room_label`` equals the value
          - ``object``: match snapshots whose ``objects_detected`` contains the value

        Returns the last `limit` matching snapshots (most recent last).
        """
        data = self._load()
        snapshots = data.get("snapshots", [])
        if not isinstance(snapshots, list):
            return []

        if limit <= 0:
            return []

        if not filter:
            return snapshots[-limit:]

        result: List[Dict[str, Any]] = []
        for snap in snapshots:
            if not isinstance(snap, dict):
                continue
            if "room" in filter:
                if snap.get("room_label") != filter["room"]:
                    continue
            if "object" in filter:
                objs = snap.get("objects_detected", [])
                if not isinstance(objs, list) or filter["object"] not in objs:
                    continue
            result.append(snap)
        return result[-limit:]

    def summarize(self, max_tokens: int) -> str:
        """Produce a text summary matching the existing
        ``PromptBuilder._format_spatial_summary`` output exactly::

            Snapshots: 5 | Rooms: kitchen, hallway | Objects: table, chair
            MANDATORY: After each look + image analysis, include spatial_memory_append ...

        Includes the MANDATORY reinforcement line that tells the LLM
        to always emit ``spatial_memory_append`` entries. This
        duplicates Rule 5.6 in the system prompt (belt-and-suspenders)
        but empirically helps weaker LLMs maintain consistent memory
        accumulation.

        The ``max_tokens`` budget is accepted for protocol compliance
        but currently unused. Future versions may use it to adaptively
        include more or fewer snapshot details.
        """
        data = self._load()
        snapshots = data.get("snapshots", [])
        rooms = list(data.get("rooms", {}).keys()) if isinstance(data.get("rooms"), dict) else []
        objects = list(data.get("object_sightings", {}).keys()) if isinstance(data.get("object_sightings"), dict) else []

        snap_count = len(snapshots) if isinstance(snapshots, list) else 0
        return (
            f"Snapshots: {snap_count}"
            f" | Rooms: {', '.join(rooms) if rooms else 'none'}"
            f" | Objects: {', '.join(objects) if objects else 'none'}"
            f"\nMANDATORY: After each look + image analysis, include spatial_memory_append in your update_nav_status call."
        )

    def export(self) -> Dict[str, Any]:
        """Return the full spatial memory data structure."""
        return self._load()

    def clear(self) -> None:
        """No-op. Per-loop lifetime is managed by the bridge — each
        nav_loop gets a fresh spatial_memory file at startup."""
        pass


__all__ = ["SpatialMemory"]
