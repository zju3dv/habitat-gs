"""Prompt asset layer — PromptFragment, PromptSpec, PromptLibrary.

Declarative, versioned, composable prompt system backed by YAML/MD
asset files on disk. Replaces the hardcoded Python f-string approach
with a structured prompt composition pipeline.

Architecture:

  PromptFragment    — a reusable rule block (e.g. "collision_recovery")
                      stored as a Markdown file under ``fragments/``
  PromptSpec        — a complete controller definition that references
                      fragments by ID and adds task-specific body;
                      stored as YAML under ``controllers/``
  PromptLibrary     — loads fragments + controllers from a root dir,
                      assembles a full system prompt via ``render()``,
                      and exposes ``register_variant`` / ``record_outcome``
                      for variant management and outcome tracking

The render pipeline:

  1. Load the PromptSpec for (task_type, nav_mode)
  2. Load each referenced fragment by ID
  3. Concatenate: preamble + fragments (in order) + controller_body
  4. Process ``{{#if var == "value"}}...{{#else}}...{{/if}}`` conditionals
  5. Replace ``${var}`` placeholders from the substitutions dict
  6. Append ``memory_bundle.render_context(spec.memory_budgets)``
  7. Return the final prompt string
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = (
    "You are a navigation agent controlling a robot in a 3D indoor environment.\n"
    "You navigate by calling tools (forward, turn, look, panorama, depth_analyze, etc.) "
    "and updating navigation status via update_nav_status.\n\n"
    "## Key Rules"
)

# Sentinel in the YAML fragments list that marks where controller_body
# is injected. Matches the legacy f-string insertion point between
# Rule 1.5 and Rule 5 in build_system_prompt.
_CONTROLLER_SENTINEL = "__controller__"


@dataclass
class PromptFragment:
    """A reusable rule block stored as a Markdown file.

    Example: ``collision_recovery`` containing Rule 7 text.
    Fragments are referenced by ID in PromptSpec.fragments lists.
    """

    id: str
    version: int = 1
    body: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    author: str = "human"
    parent_version: Optional[int] = None
    created_at: str = ""


@dataclass
class PromptSpec:
    """A complete controller definition — declarative, serializable,
    versionable. Stored as YAML under ``controllers/``."""

    # Identity
    name: str                    # e.g. "pointnav_mapless"
    task_type: str
    nav_mode: str
    version: int = 1

    # Composition
    fragments: List[str] = field(default_factory=list)  # fragment IDs
    controller_body: str = ""     # task-specific rules (Rules 2-4)
    variables: List[str] = field(default_factory=list)

    # Memory budget for render-time injection
    memory_budgets: Dict[str, int] = field(default_factory=dict)

    # Authorship / provenance
    author: str = "human"
    parent_version: Optional[int] = None
    rationale: Optional[str] = None
    created_at: str = ""

    # Outcome tracking (populated by record_outcome)
    success_rate_history: List[Tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Conditional processor — minimal {{#if}} / {{#else}} / {{/if}}
# ---------------------------------------------------------------------------

_IF_RE = re.compile(
    r"\{\{#if\s+(\w+)\s*==\s*\"([^\"]*)\"\s*\}\}",
    re.DOTALL,
)


def _process_conditionals(text: str, context: Dict[str, str]) -> str:
    """Process ``{{#if var == "value"}}...{{#else}}...{{/if}}`` blocks.

    Only supports simple equality checks against string values. This
    is intentionally minimal — the only conditional in the current
    prompt set is ``nav_mode == "mapless"`` / ``"navmesh"``. If more
    complex logic is ever needed, switch to Jinja2.
    """
    result = text
    # Process from innermost out (simple — no nesting expected)
    while "{{#if" in result:
        match = _IF_RE.search(result)
        if not match:
            break
        var_name = match.group(1)
        var_value = match.group(2)
        start = match.start()

        # Find the matching {{/if}}
        endif_pos = result.find("{{/if}}", match.end())
        if endif_pos == -1:
            break  # malformed — leave as-is

        block = result[match.end():endif_pos]

        # Split on {{#else}} if present
        else_pos = block.find("{{#else}}")
        if else_pos != -1:
            if_branch = block[:else_pos]
            else_branch = block[else_pos + len("{{#else}}"):]
        else:
            if_branch = block
            else_branch = ""

        # Evaluate
        actual = context.get(var_name, "")
        chosen = if_branch if actual == var_value else else_branch

        result = result[:start] + chosen.strip() + result[endif_pos + len("{{/if}}"):]

    return result


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------


class PromptLibrary:
    """Central registry for prompt assets.

    Directory layout::

        root/
        ├── fragments/
        │   ├── tool_call_model.md
        │   ├── collision_recovery.md
        │   └── ...
        └── controllers/
            ├── pointnav_mapless.yaml
            ├── pointnav_navmesh.yaml
            └── ...
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._fragment_cache: Dict[str, PromptFragment] = {}
        self._controller_cache: Dict[str, PromptSpec] = {}

    # ── Loading ──────────────────────────────────────────────────

    def load_fragment(self, fragment_id: str) -> PromptFragment:
        """Load a fragment from ``fragments/{id}.md``."""
        if fragment_id in self._fragment_cache:
            return self._fragment_cache[fragment_id]

        path = self.root / "fragments" / f"{fragment_id}.md"
        if not path.is_file():
            raise FileNotFoundError(f"Fragment not found: {path}")

        body = path.read_text(encoding="utf-8").strip()
        frag = PromptFragment(id=fragment_id, body=body)
        self._fragment_cache[fragment_id] = frag
        return frag

    def load_controller(
        self, task_type: str, nav_mode: str
    ) -> PromptSpec:
        """Load a controller spec from ``controllers/{task_type}_{nav_mode}.yaml``."""
        key = f"{task_type}_{nav_mode}"
        if key in self._controller_cache:
            return self._controller_cache[key]

        path = self.root / "controllers" / f"{key}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Controller not found: {path}")

        # Use safe YAML loading (PyYAML or fallback to json-ish parsing)
        data = self._load_yaml(path)
        spec = PromptSpec(
            name=data.get("name", key),
            task_type=data.get("task_type", task_type),
            nav_mode=data.get("nav_mode", nav_mode),
            version=data.get("version", 1),
            fragments=data.get("fragments", []),
            controller_body=data.get("controller_body", ""),
            variables=data.get("variables", []),
            memory_budgets=data.get("memory_budgets", {}),
            author=data.get("author", "human"),
            parent_version=data.get("parent_version"),
            rationale=data.get("rationale"),
            created_at=data.get("created_at", ""),
        )
        self._controller_cache[key] = spec
        return spec

    # ── Rendering ────────────────────────────────────────────────

    def render(
        self,
        spec: PromptSpec,
        substitutions: Dict[str, str],
        nav_mode: str,
        memory_bundle: Optional[Any] = None,
    ) -> str:
        """Assemble the full system prompt from spec + fragments.

        Pipeline:
          1. Start with preamble
          2. Load and concatenate each referenced fragment
          3. Append controller_body
          4. Process {{#if}} conditionals using nav_mode
          5. Replace ${var} placeholders
          6. Append memory context from memory_bundle
        """
        parts: List[str] = [_SYSTEM_PREAMBLE]
        controller_injected = False

        # Fragments + controller body (in order from spec.fragments).
        # The special sentinel __controller__ marks where controller_body
        # is inserted. This matches the legacy f-string structure where
        # Rules 2-4 (task-specific) appear BETWEEN Rule 1.5 and Rule 5.
        for frag_id in spec.fragments:
            if frag_id == _CONTROLLER_SENTINEL:
                if spec.controller_body:
                    # Add trailing \n to match the legacy .txt file behavior:
                    # the original files ended with \n, which combined with
                    # the f-string's \n\n produced 2 blank lines before Rule 5.
                    parts.append(spec.controller_body.strip() + "\n")
                controller_injected = True
                continue
            try:
                frag = self.load_fragment(frag_id)
                parts.append(frag.body)
            except FileNotFoundError:
                parts.append(f"<!-- missing fragment: {frag_id} -->")

        # Fallback: if no sentinel was present, append controller at end
        if not controller_injected and spec.controller_body:
            parts.append(spec.controller_body.strip())

        # Assemble
        text = "\n\n".join(parts)

        # Conditionals
        context = {"nav_mode": nav_mode}
        context.update(substitutions)
        text = _process_conditionals(text, context)

        # Variable substitution
        for var, val in substitutions.items():
            text = text.replace(f"${{{var}}}", str(val or ""))

        # Memory context
        if memory_bundle is not None and spec.memory_budgets:
            mem_text = memory_bundle.render_context(spec.memory_budgets)
            if mem_text:
                text = text + "\n\n" + mem_text

        return text

    # ── Variant management ────────────────────────────────────────

    def register_variant(self, spec: PromptSpec) -> Path:
        """Save a new prompt variant to disk. Returns the written path."""
        variants_dir = self.root / "controllers" / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{spec.name}_v{spec.version}.yaml"
        path = variants_dir / filename
        self._write_yaml(path, {
            "name": spec.name,
            "task_type": spec.task_type,
            "nav_mode": spec.nav_mode,
            "version": spec.version,
            "fragments": spec.fragments,
            "controller_body": spec.controller_body,
            "variables": spec.variables,
            "memory_budgets": spec.memory_budgets,
            "author": spec.author,
            "parent_version": spec.parent_version,
            "rationale": spec.rationale,
            "created_at": spec.created_at or datetime.now(timezone.utc).isoformat(),
        })
        return path

    def record_outcome(
        self,
        spec_name: str,
        version: int,
        run_id: str,
        success_rate: float,
    ) -> None:
        """Append a benchmark outcome to the spec's history file."""
        outcomes_dir = self.root / "controllers" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        path = outcomes_dir / f"{spec_name}.jsonl"
        entry = {
            "version": version,
            "run_id": run_id,
            "success_rate": success_rate,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── YAML helpers ─────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML file. Tries PyYAML first, falls back to a
        simple key-value parser for environments without PyYAML."""
        text = path.read_text(encoding="utf-8")
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            pass
        # Minimal fallback: parse simple YAML subset
        # (good enough for our flat-ish controller specs)
        return _simple_yaml_parse(text)

    @staticmethod
    def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
        """Write YAML file."""
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
        except ImportError:
            # Fallback: write as JSON (valid YAML subset)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for flat controller specs.

    Handles: scalar values, simple lists (``- item``), and
    multi-line strings (``key: |``). NOT a general YAML parser.
    """
    result: Dict[str, Any] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if value.startswith("[") and value.endswith("]"):
                # Inline list: [] or [item1, item2]
                # Codex P2: the previous code fell through to the scalar
                # branch and stored "[]" as a string, violating the
                # PromptSpec.variables type contract (List[str]).
                inner = value[1:-1].strip()
                if not inner:
                    result[key] = []
                else:
                    result[key] = [
                        x.strip().strip('"').strip("'")
                        for x in inner.split(",")
                        if x.strip()
                    ]
                i += 1
                continue

            if value == "|":
                # Multi-line block scalar
                block_lines: List[str] = []
                i += 1
                while i < len(lines):
                    bline = lines[i]
                    if bline and not bline[0].isspace():
                        break
                    block_lines.append(bline)
                    i += 1
                # Dedent
                if block_lines:
                    indent = len(block_lines[0]) - len(block_lines[0].lstrip())
                    result[key] = "\n".join(
                        bl[indent:] if len(bl) > indent else bl
                        for bl in block_lines
                    ).rstrip("\n")
                else:
                    result[key] = ""
                continue

            if value == "":
                # Could be a list or nested dict — check next lines
                items: List[str] = []
                nested: Dict[str, Any] = {}
                i += 1
                while i < len(lines):
                    raw_line = lines[i]
                    nline = raw_line.strip()
                    if not nline:
                        i += 1
                        continue
                    # Still indented? Part of this block
                    if raw_line[0].isspace():
                        if nline.startswith("- "):
                            items.append(nline[2:].strip())
                        elif ":" in nline:
                            # Nested key: value
                            nk, _, nv = nline.partition(":")
                            nv = nv.strip()
                            if nv.isdigit():
                                nested[nk.strip()] = int(nv)
                            elif nv.startswith('"') and nv.endswith('"'):
                                nested[nk.strip()] = nv[1:-1]
                            else:
                                nested[nk.strip()] = nv
                        i += 1
                    else:
                        break  # next top-level key
                if items:
                    result[key] = items
                elif nested:
                    result[key] = nested
                continue

            # Simple scalar
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.isdigit():
                result[key] = int(value)
                i += 1
                continue
            elif value in ("true", "True"):
                result[key] = True
                i += 1
                continue
            elif value in ("false", "False"):
                result[key] = False
                i += 1
                continue
            result[key] = value

        i += 1

    return result


__all__ = [
    "PromptFragment",
    "PromptSpec",
    "PromptLibrary",
]
