"""Phase 3 PR 1 — unit tests for Memory protocol + MemoryBundle + SpatialMemory.

Tests cover:
  - MemoryBundle registration, lookup, render_context, export_all
  - SpatialMemory loading (missing file, malformed, valid),
    summarize, query (by room, by object), export, add no-op
  - Protocol runtime check (isinstance)
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from habitat_agent.memory.base import Memory, MemoryBundle  # noqa: E402
from habitat_agent.memory.spatial import SpatialMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_SPATIAL = {
    "task_id": "navloop-test",
    "grid_resolution_m": 0.5,
    "snapshots": [
        {
            "heading_deg": 90,
            "scene_description": "Office with desk",
            "room_label": "office",
            "objects_detected": ["desk", "chair", "monitor"],
            "ts": "2026-04-10T01:00:00Z",
        },
        {
            "heading_deg": 180,
            "scene_description": "Kitchen area",
            "room_label": "kitchen",
            "objects_detected": ["table", "microwave"],
            "ts": "2026-04-10T01:01:00Z",
        },
        {
            "heading_deg": 270,
            "scene_description": "Back in office",
            "room_label": "office",
            "objects_detected": ["desk", "bookshelf"],
            "ts": "2026-04-10T01:02:00Z",
        },
    ],
    "rooms": {
        "office": {"visit_count": 2, "first_seen": "2026-04-10T01:00:00Z", "last_seen": "2026-04-10T01:02:00Z"},
        "kitchen": {"visit_count": 1, "first_seen": "2026-04-10T01:01:00Z", "last_seen": "2026-04-10T01:01:00Z"},
    },
    "object_sightings": {
        "desk": {"count": 2, "last_seen": "2026-04-10T01:02:00Z"},
        "chair": {"count": 1, "last_seen": "2026-04-10T01:00:00Z"},
        "monitor": {"count": 1, "last_seen": "2026-04-10T01:00:00Z"},
        "table": {"count": 1, "last_seen": "2026-04-10T01:01:00Z"},
        "microwave": {"count": 1, "last_seen": "2026-04-10T01:01:00Z"},
        "bookshelf": {"count": 1, "last_seen": "2026-04-10T01:02:00Z"},
    },
}


@pytest.fixture
def spatial_file(tmp_path):
    """Write a sample spatial_memory.json and return the path."""
    path = tmp_path / "spatial_memory.json"
    path.write_text(json.dumps(_SAMPLE_SPATIAL, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# MemoryBundle
# ---------------------------------------------------------------------------


def test_bundle_register_and_get(spatial_file):
    bundle = MemoryBundle()
    sm = SpatialMemory(spatial_file)
    bundle.register(sm)

    assert bundle.get("spatial") is sm
    assert bundle.get("nonexistent") is None
    assert bundle.list_names() == ["spatial"]


def test_bundle_render_empty():
    bundle = MemoryBundle()
    result = bundle.render_context({"spatial": 500})
    assert result == ""


def test_bundle_render_with_spatial(spatial_file):
    bundle = MemoryBundle()
    bundle.register(SpatialMemory(spatial_file))

    result = bundle.render_context({"spatial": 500})
    assert "## spatial" in result
    assert "Snapshots: 3" in result
    assert "office" in result
    assert "kitchen" in result
    assert "desk" in result


def test_bundle_render_respects_budget_keys(spatial_file):
    """Only memories listed in the budgets dict are rendered."""
    bundle = MemoryBundle()
    bundle.register(SpatialMemory(spatial_file))

    # Ask for "episodic" which doesn't exist — should produce empty
    result = bundle.render_context({"episodic": 300})
    assert result == ""

    # Ask for both — only spatial produces output
    result = bundle.render_context({"spatial": 500, "episodic": 300})
    assert "## spatial" in result
    assert "## episodic" not in result


def test_bundle_render_multiple_memories(spatial_file):
    """When multiple memories produce output, they're joined with
    double-newline separators and each gets a ## header."""

    # Create a second memory that always returns something
    class DummyMemory:
        name = "dummy"
        persistence = "durable"
        def add(self, entry): pass
        def query(self, filter=None, limit=10): return []
        def summarize(self, max_tokens): return "dummy data here"
        def export(self): return {}
        def clear(self): pass

    bundle = MemoryBundle()
    bundle.register(SpatialMemory(spatial_file))
    bundle.register(DummyMemory())

    result = bundle.render_context({"spatial": 500, "dummy": 100})
    assert "## spatial" in result
    assert "## dummy" in result
    assert "dummy data here" in result
    # Sections separated by double newline
    assert "\n\n" in result


def test_bundle_export_all(spatial_file):
    bundle = MemoryBundle()
    bundle.register(SpatialMemory(spatial_file))
    exported = bundle.export_all()
    assert "spatial" in exported
    assert exported["spatial"]["snapshots"] == _SAMPLE_SPATIAL["snapshots"]


def test_bundle_add_to(spatial_file):
    """add_to delegates to the named memory's add method."""
    bundle = MemoryBundle()
    sm = SpatialMemory(spatial_file)
    bundle.register(sm)
    # SpatialMemory.add is a no-op, so this should not raise
    bundle.add_to("spatial", {"heading_deg": 0, "scene_description": "test"})
    # add_to for missing memory is also silent
    bundle.add_to("nonexistent", {"x": 1})


# ---------------------------------------------------------------------------
# SpatialMemory
# ---------------------------------------------------------------------------


def test_spatial_load_missing_file():
    sm = SpatialMemory("/nonexistent/path/spatial_memory.json")
    data = sm._load()
    assert data == {"snapshots": [], "rooms": {}, "object_sightings": {}}


def test_spatial_load_empty_path():
    sm = SpatialMemory("")
    data = sm._load()
    assert data == {"snapshots": [], "rooms": {}, "object_sightings": {}}


def test_spatial_load_malformed_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all {{{", encoding="utf-8")
    sm = SpatialMemory(str(bad))
    data = sm._load()
    assert data == {"snapshots": [], "rooms": {}, "object_sightings": {}}


def test_spatial_load_non_dict_json(tmp_path):
    """If the file contains a JSON array instead of object, treat as empty."""
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    sm = SpatialMemory(str(path))
    data = sm._load()
    assert data == {"snapshots": [], "rooms": {}, "object_sightings": {}}


def test_spatial_summarize_empty():
    sm = SpatialMemory("/nonexistent")
    summary = sm.summarize(max_tokens=500)
    assert "Snapshots: 0 | Rooms: none | Objects: none" in summary
    assert "MANDATORY" in summary


def test_spatial_summarize_with_data(spatial_file):
    sm = SpatialMemory(spatial_file)
    summary = sm.summarize(max_tokens=500)
    assert "Snapshots: 3" in summary
    assert "office" in summary
    assert "kitchen" in summary
    assert "desk" in summary
    assert "microwave" in summary


def test_spatial_query_no_filter(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query()
    assert len(results) == 3
    assert results[0]["room_label"] == "office"
    assert results[1]["room_label"] == "kitchen"
    assert results[2]["room_label"] == "office"


def test_spatial_query_with_limit(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query(limit=2)
    assert len(results) == 2
    # Returns last 2 (most recent)
    assert results[0]["room_label"] == "kitchen"
    assert results[1]["room_label"] == "office"


def test_spatial_query_by_room(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query(filter={"room": "kitchen"})
    assert len(results) == 1
    assert results[0]["room_label"] == "kitchen"
    assert "table" in results[0]["objects_detected"]


def test_spatial_query_by_object(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query(filter={"object": "desk"})
    assert len(results) == 2  # desk appears in 2 snapshots
    for r in results:
        assert "desk" in r["objects_detected"]


def test_spatial_query_by_room_and_object(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query(filter={"room": "office", "object": "bookshelf"})
    assert len(results) == 1
    assert results[0]["room_label"] == "office"
    assert "bookshelf" in results[0]["objects_detected"]


def test_spatial_query_limit_zero_returns_empty(spatial_file):
    """Codex P3: limit=0 must return [] not the full list.
    Python's [-0:] == [0:] == full slice, so without a guard
    limit=0 would violate the 'last N entries' contract."""
    sm = SpatialMemory(spatial_file)
    assert sm.query(limit=0) == []
    assert sm.query(filter={"room": "office"}, limit=0) == []


def test_spatial_query_negative_limit_returns_empty(spatial_file):
    sm = SpatialMemory(spatial_file)
    assert sm.query(limit=-1) == []


def test_spatial_query_no_match(spatial_file):
    sm = SpatialMemory(spatial_file)
    results = sm.query(filter={"room": "bathroom"})
    assert results == []


def test_spatial_export(spatial_file):
    sm = SpatialMemory(spatial_file)
    exported = sm.export()
    assert exported["task_id"] == "navloop-test"
    assert len(exported["snapshots"]) == 3


def test_spatial_add_is_noop(spatial_file):
    """add() is a no-op because the bridge handles writes."""
    sm = SpatialMemory(spatial_file)
    sm.add({"heading_deg": 0, "scene_description": "test"})
    # File should not be modified
    with open(spatial_file, "r") as f:
        data = json.load(f)
    assert len(data["snapshots"]) == 3  # unchanged


def test_spatial_load_fallback_returns_independent_structures():
    """Codex P2 regression lock — shallow copy bug.

    Previously _load() used dict(_EMPTY_SPATIAL) which shallow-copied
    the top-level dict but shared the nested lists/dicts. If a caller
    mutated export() output, the next _load() fallback would contain
    phantom data. Each fallback return must be fully independent."""
    sm = SpatialMemory("/nonexistent")

    a = sm._load()
    b = sm._load()

    # Mutate a's nested structures
    a["snapshots"].append({"heading_deg": 0})
    a["rooms"]["phantom_room"] = {"visit_count": 1}
    a["object_sightings"]["phantom_object"] = {"count": 1}

    # b must NOT see the mutations
    assert b["snapshots"] == [], (
        f"shallow copy bug: b['snapshots'] = {b['snapshots']} "
        f"(should be empty, was contaminated by mutation of a)"
    )
    assert b["rooms"] == {}
    assert b["object_sightings"] == {}


def test_spatial_clear_is_noop(spatial_file):
    sm = SpatialMemory(spatial_file)
    sm.clear()
    # File should still exist and be unchanged
    assert os.path.isfile(spatial_file)


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------


def test_spatial_memory_satisfies_protocol():
    sm = SpatialMemory("/nonexistent")
    assert isinstance(sm, Memory)


def test_non_memory_fails_protocol():
    class NotAMemory:
        pass
    assert not isinstance(NotAMemory(), Memory)
