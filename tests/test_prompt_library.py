"""Phase 4 PR 1 — unit tests for PromptFragment, PromptSpec, PromptLibrary.

Tests cover:
  - Fragment loading from .md files
  - Controller loading from .yaml files
  - render() pipeline: preamble + fragments + controller_body + conditionals + vars
  - {{#if}} conditional processing
  - register_variant writes to disk
  - record_outcome appends JSONL
  - Error handling for missing files
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from habitat_agent.prompts.spec import (  # noqa: E402
    PromptFragment,
    PromptLibrary,
    PromptSpec,
    _process_conditionals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def library_root(tmp_path):
    """Create a minimal PromptLibrary directory structure with
    sample fragments and a controller."""
    frags = tmp_path / "fragments"
    frags.mkdir()
    ctrls = tmp_path / "controllers"
    ctrls.mkdir()

    # Fragment: tool_call_model
    (frags / "tool_call_model.md").write_text(
        "**Rule 1 — Tool-based interaction.**\n"
        "{{#if nav_mode == \"mapless\"}}\n"
        "You are in MAPLESS mode.\n"
        "{{#else}}\n"
        "You have full navmesh tools.\n"
        "{{/if}}\n",
        encoding="utf-8",
    )

    # Fragment: terminal_handling
    (frags / "terminal_handling.md").write_text(
        "**Rule 8 — Terminal handling.**\n"
        "When done: export_video then update_nav_status.\n",
        encoding="utf-8",
    )

    # Controller: pointnav_mapless
    (ctrls / "pointnav_mapless.yaml").write_text(
        "name: pointnav_mapless\n"
        "task_type: pointnav\n"
        "nav_mode: mapless\n"
        "version: 1\n"
        "author: human\n"
        "fragments:\n"
        "  - tool_call_model\n"
        "  - terminal_handling\n"
        "memory_budgets:\n"
        "  spatial: 500\n"
        "variables:\n"
        "  - goal_desc\n"
        "controller_body: |\n"
        "  **Rule 2 — Decompose.**\n"
        "  Goal: ${goal_desc}\n",
        encoding="utf-8",
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Fragment loading
# ---------------------------------------------------------------------------


def test_load_fragment(library_root):
    lib = PromptLibrary(library_root)
    frag = lib.load_fragment("tool_call_model")
    assert frag.id == "tool_call_model"
    assert "Rule 1" in frag.body
    assert "{{#if" in frag.body  # conditional not yet processed


def test_load_fragment_cached(library_root):
    lib = PromptLibrary(library_root)
    f1 = lib.load_fragment("tool_call_model")
    f2 = lib.load_fragment("tool_call_model")
    assert f1 is f2


def test_load_fragment_missing(library_root):
    lib = PromptLibrary(library_root)
    with pytest.raises(FileNotFoundError, match="Fragment not found"):
        lib.load_fragment("nonexistent")


# ---------------------------------------------------------------------------
# Controller loading
# ---------------------------------------------------------------------------


def test_load_controller(library_root):
    lib = PromptLibrary(library_root)
    spec = lib.load_controller("pointnav", "mapless")
    assert spec.name == "pointnav_mapless"
    assert spec.task_type == "pointnav"
    assert spec.nav_mode == "mapless"
    assert spec.version == 1
    assert "tool_call_model" in spec.fragments
    assert "terminal_handling" in spec.fragments
    assert "goal_desc" in spec.variables
    assert spec.memory_budgets == {"spatial": 500}
    assert "Rule 2" in spec.controller_body


def test_load_controller_missing(library_root):
    lib = PromptLibrary(library_root)
    with pytest.raises(FileNotFoundError, match="Controller not found"):
        lib.load_controller("eqa", "navmesh")


# ---------------------------------------------------------------------------
# Conditional processing
# ---------------------------------------------------------------------------


def test_conditionals_if_true():
    text = '{{#if nav_mode == "mapless"}}MAPLESS MODE{{#else}}NAVMESH MODE{{/if}}'
    result = _process_conditionals(text, {"nav_mode": "mapless"})
    assert result == "MAPLESS MODE"


def test_conditionals_if_false():
    text = '{{#if nav_mode == "mapless"}}MAPLESS MODE{{#else}}NAVMESH MODE{{/if}}'
    result = _process_conditionals(text, {"nav_mode": "navmesh"})
    assert result == "NAVMESH MODE"


def test_conditionals_no_else():
    text = '{{#if nav_mode == "mapless"}}MAPLESS ONLY{{/if}}'
    result = _process_conditionals(text, {"nav_mode": "mapless"})
    assert result == "MAPLESS ONLY"

    result2 = _process_conditionals(text, {"nav_mode": "navmesh"})
    assert "MAPLESS" not in result2


def test_conditionals_missing_variable():
    text = '{{#if nav_mode == "mapless"}}YES{{#else}}NO{{/if}}'
    result = _process_conditionals(text, {})  # nav_mode not in context
    assert result == "NO"


def test_conditionals_no_markers():
    text = "Plain text without any conditionals."
    result = _process_conditionals(text, {"nav_mode": "mapless"})
    assert result == text


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_assembles_preamble_fragments_controller(library_root):
    lib = PromptLibrary(library_root)
    spec = lib.load_controller("pointnav", "mapless")
    result = lib.render(
        spec,
        substitutions={"goal_desc": "Find the kitchen"},
        nav_mode="mapless",
    )

    # Preamble present
    assert "You are a navigation agent" in result
    # Fragment content present (conditional processed for mapless)
    assert "MAPLESS mode" in result
    assert "navmesh tools" not in result  # else branch NOT included
    # Terminal handling fragment present
    assert "Rule 8" in result
    # Controller body present with substitution
    assert "Rule 2" in result
    assert "Find the kitchen" in result
    assert "${goal_desc}" not in result  # placeholder replaced


def test_render_navmesh_conditional(library_root):
    lib = PromptLibrary(library_root)
    spec = lib.load_controller("pointnav", "mapless")
    # Render with navmesh mode — conditional should pick else branch
    result = lib.render(
        spec,
        substitutions={"goal_desc": "Go there"},
        nav_mode="navmesh",
    )
    assert "navmesh tools" in result
    assert "MAPLESS mode" not in result


def test_render_with_memory_bundle(library_root):
    """render() appends memory context when memory_bundle is provided."""
    from habitat_agent.memory.base import MemoryBundle

    class FakeMemory:
        name = "spatial"
        persistence = "per-loop"
        def add(self, e): pass
        def query(self, f=None, l=10): return []
        def summarize(self, m): return "Snapshots: 5 | Rooms: office | Objects: desk"
        def export(self): return {}
        def clear(self): pass

    bundle = MemoryBundle()
    bundle.register(FakeMemory())

    lib = PromptLibrary(library_root)
    spec = lib.load_controller("pointnav", "mapless")
    result = lib.render(
        spec,
        substitutions={"goal_desc": "test"},
        nav_mode="mapless",
        memory_bundle=bundle,
    )
    assert "## spatial" in result
    assert "Snapshots: 5" in result


def test_render_without_memory_bundle(library_root):
    lib = PromptLibrary(library_root)
    spec = lib.load_controller("pointnav", "mapless")
    result = lib.render(
        spec,
        substitutions={"goal_desc": "test"},
        nav_mode="mapless",
        memory_bundle=None,
    )
    # No memory section appended
    assert "## spatial" not in result


def test_render_missing_fragment_graceful(library_root):
    """If a referenced fragment doesn't exist, render inserts a
    placeholder comment instead of crashing."""
    lib = PromptLibrary(library_root)
    spec = PromptSpec(
        name="test",
        task_type="pointnav",
        nav_mode="mapless",
        fragments=["nonexistent_fragment"],
        controller_body="body here",
    )
    result = lib.render(spec, {}, nav_mode="mapless")
    assert "<!-- missing fragment: nonexistent_fragment -->" in result
    assert "body here" in result


# ---------------------------------------------------------------------------
# register_variant + record_outcome
# ---------------------------------------------------------------------------


def test_simple_yaml_parse_inline_empty_list():
    """Codex P2 regression lock: _simple_yaml_parse must parse inline
    YAML lists like `variables: []` as Python lists, not as the string
    '[]'. PyYAML is not installed in this environment, so the fallback
    parser IS used in production. A string '[]' breaks register_variant
    serialization and any future caller that iterates spec.variables."""
    from habitat_agent.prompts.spec import _simple_yaml_parse

    # Empty inline list
    result = _simple_yaml_parse("variables: []")
    assert isinstance(result.get("variables"), list), (
        f"variables: [] parsed as {type(result.get('variables')).__name__!r} "
        f"not list. Got: {result.get('variables')!r}"
    )
    assert result["variables"] == []


def test_simple_yaml_parse_inline_list_with_items():
    """Same fix: inline list with items should also parse correctly."""
    from habitat_agent.prompts.spec import _simple_yaml_parse

    result = _simple_yaml_parse('tags: [safety, execution]')
    assert result.get("tags") == ["safety", "execution"], (
        f"tags: [safety, execution] parsed incorrectly: {result.get('tags')!r}"
    )


def test_load_controller_variables_field_is_list():
    """End-to-end: loading pointnav_navmesh.yaml must yield a spec
    with spec.variables as a list (not the string '[]')."""
    lib = PromptLibrary(Path(_PROMPTS_ROOT))
    spec = lib.load_controller("pointnav", "navmesh")
    assert isinstance(spec.variables, list), (
        f"spec.variables for pointnav_navmesh is {type(spec.variables).__name__}, "
        f"expected list (register_variant would serialise it as {spec.variables!r})."
    )


def test_register_variant_writes_file(library_root):
    lib = PromptLibrary(library_root)
    spec = PromptSpec(
        name="pointnav_mapless",
        task_type="pointnav",
        nav_mode="mapless",
        version=2,
        author="test_agent",
        parent_version=1,
        rationale="Improved collision recovery",
        controller_body="new rules here",
        fragments=["tool_call_model"],
    )
    path = lib.register_variant(spec)
    assert path.exists()
    assert "pointnav_mapless_v2" in path.name

    # Read back and verify
    content = path.read_text(encoding="utf-8")
    assert "pointnav_mapless" in content
    assert "test_agent" in content


def test_record_outcome_appends_jsonl(library_root):
    lib = PromptLibrary(library_root)
    lib.record_outcome("pointnav_mapless", version=1, run_id="bench-001", success_rate=0.85)
    lib.record_outcome("pointnav_mapless", version=1, run_id="bench-002", success_rate=0.72)

    outcomes_file = library_root / "controllers" / "outcomes" / "pointnav_mapless.jsonl"
    assert outcomes_file.exists()

    lines = outcomes_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["version"] == 1
    assert entry1["run_id"] == "bench-001"
    assert entry1["success_rate"] == 0.85
    assert "ts" in entry1

    entry2 = json.loads(lines[1])
    assert entry2["run_id"] == "bench-002"


# ---------------------------------------------------------------------------
# Render-equivalence tests — new YAML assets must produce output that
# matches the legacy PromptBuilder when given the same nav_status input.
# These are the critical safety net before switching the live prompt path.
# ---------------------------------------------------------------------------

import os as _os
_PROMPTS_ROOT = _os.path.join(_TOOLS_DIR, "habitat_agent", "prompts")


def _make_nav_status(task_type="pointnav", nav_mode="mapless",
                     goal_description="Navigate to the kitchen",
                     reference_image=""):
    return {
        "task_id": "navloop-test",
        "task_type": task_type,
        "nav_mode": nav_mode,
        "status": "in_progress",
        "nav_phase": "navigating",
        "total_steps": 5,
        "collisions": 0,
        "state_version": 3,
        "session_id": "sess-test",
        "goal_description": goal_description,
        "goal_position": None,
        "goal_type": "instruction",
        "has_navmesh": nav_mode == "navmesh",
        "is_gaussian": False,
        "reference_image": reference_image,
        "action_history": [],
        "substeps": [],
        "current_substep_index": 0,
        "spatial_memory_file": "",
        "euclidean_distance_to_goal": 3.5,
        "goal_direction_deg": 45.0,
    }


def _normalise(text):
    """Normalise whitespace for comparison — collapse multiple blanks
    and strip trailing spaces per line."""
    lines = []
    for line in text.split("\n"):
        lines.append(line.rstrip())
    # Collapse 3+ consecutive blank lines to 2
    result = []
    blanks = 0
    for line in lines:
        if line == "":
            blanks += 1
        else:
            blanks = 0
        if blanks <= 2:
            result.append(line)
    return "\n".join(result).strip()


def _legacy_prompt(task_type, nav_mode):
    from habitat_agent.prompts.legacy_builder import PromptBuilder
    pb = PromptBuilder(nav_mode=nav_mode)
    nav_status = _make_nav_status(task_type=task_type, nav_mode=nav_mode)
    return pb.build_system_prompt(nav_status)


def _library_prompt(task_type, nav_mode):
    lib = PromptLibrary(Path(_PROMPTS_ROOT))
    spec = lib.load_controller(task_type, nav_mode)
    nav_status = _make_nav_status(task_type=task_type, nav_mode=nav_mode)
    subs = {
        "goal_desc": nav_status["goal_description"],
        "goal_description": nav_status["goal_description"],
        "reference_image": nav_status.get("reference_image", ""),
        "task_type": task_type,
        "nav_mode": nav_mode,
    }
    return lib.render(spec, subs, nav_mode)


_TASK_SPECIFIC_MARKERS = {
    # Each tuple: (required_phrase_in_output,)
    "pointnav": ("euclidean_distance_to_goal",),
    "objectnav": ("ObjectNav", "target object"),
    "imagenav": ("ImageNav", "reference image"),
    "instruction_following": ("Instruction Following", "goal_description"),
    "eqa": ("EQA", "answer"),
}

_NAV_MODE_MARKERS = {
    "mapless": ("MAPLESS",),
    "navmesh": ("navmesh",),
}


@pytest.mark.parametrize("task_type,nav_mode", [
    ("pointnav", "mapless"),
    ("pointnav", "navmesh"),
    ("objectnav", "mapless"),
    ("objectnav", "navmesh"),
    ("imagenav", "mapless"),
    ("imagenav", "navmesh"),
    ("instruction_following", "mapless"),
    ("instruction_following", "navmesh"),
    ("eqa", "mapless"),
    ("eqa", "navmesh"),
])
def test_render_structural_properties(task_type, nav_mode):
    """Codex P2: after PR 3 deleted the legacy .txt files, the old
    byte-for-byte render-equivalence comparison no longer makes sense
    (both sides now go through PromptLibrary). Replaced with structural
    property checks that catch YAML regressions regardless of code path:

      - Shared rules (1, 1.5, 5-8) must always be present
      - Nav-mode-specific content must appear
      - Task-type-specific content (Rules 2-4) must appear
      - No 'No specific controller' generic fallback string

    If someone accidentally deletes a fragment or corrupts a YAML,
    these checks will fail even when both sides use the same code path.
    """
    prompt = _normalise(_library_prompt(task_type, nav_mode))

    # ── Shared rules (fragments) ─────────────────────────────────
    shared_checks = [
        ("## Key Rules", "preamble missing"),
        ("Rule 1", "tool_call_model fragment missing"),
        ("Rule 1.5", "movement magnitude rule missing"),
        ("Rule 5", "observe_cycle fragment missing"),
        ("Rule 5.5", "structured_reasoning fragment missing"),
        ("Rule 5.6", "spatial_memory fragment missing"),
        ("Rule 6", "state_updates fragment missing"),
        ("Rule 7", "collision_recovery fragment missing"),
        ("Rule 8", "terminal_handling fragment missing"),
    ]
    for marker, msg in shared_checks:
        assert marker in prompt, (
            f"[{task_type}_{nav_mode}] {msg}: {marker!r} not found"
        )

    # ── Nav-mode-specific content ─────────────────────────────────
    for marker in _NAV_MODE_MARKERS[nav_mode]:
        assert marker.lower() in prompt.lower(), (
            f"[{task_type}_{nav_mode}] nav_mode content missing: {marker!r}"
        )

    # ── Task-specific content (Rules 2-4 from controller YAML) ────
    for marker in _TASK_SPECIFIC_MARKERS[task_type]:
        assert marker.lower() in prompt.lower(), (
            f"[{task_type}_{nav_mode}] task-specific Rule 2-4 content missing: "
            f"{marker!r}. Check controllers/{task_type}_{nav_mode}.yaml"
        )

    # ── Must NOT be the generic fallback string ───────────────────
    assert "No specific controller" not in prompt, (
        f"[{task_type}_{nav_mode}] prompt contains generic fallback text — "
        "the YAML asset failed to load and _load_controller was used instead"
    )
