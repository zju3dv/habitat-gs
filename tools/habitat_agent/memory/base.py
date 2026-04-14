"""Memory abstraction layer — `Memory` protocol + `MemoryBundle`.

Phase 3 of the architecture refactor. The goal is to give agents a
uniform, pluggable interface for reading / summarizing / querying
memories, so that:

  1. PromptBuilder can call `memory_bundle.render_context(budgets)`
     instead of inline `json.load` + ad-hoc formatting.
  2. Memory contents can be inspected for evaluation and analysis.
  3. New memory types (episodic, procedural, reference) can be added
     by implementing the `Memory` protocol and registering into a
     `MemoryBundle`, without touching the agent or prompt code.

Phase 3 only introduces `SpatialMemory` — the existing
spatial_memory_*.json wrapper. No new memory types are added.

Architecture:

  Memory (Protocol)   — what every memory type must implement
  MemoryBundle        — collection of memories for a given agent context
  SpatialMemory       — concrete implementation wrapping spatial_memory.json
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Abstract memory interface.

    All memory types (spatial, episodic, procedural, reference) implement
    this protocol so they can be mixed in a MemoryBundle with uniform
    access. The protocol is intentionally minimal — each method has a
    clear purpose, and implementations can no-op methods that don't
    apply (e.g. SpatialMemory.add is a no-op because the bridge handles
    writes).

    Lifecycle scopes:
      - "per-loop"    : cleared when a nav_loop ends (e.g. spatial memory)
      - "per-session" : persists across loops in the same session
      - "durable"     : persists across sessions (e.g. episodic memory, future)
    """

    name: str
    persistence: str  # "per-loop" | "per-session" | "durable"

    def add(self, entry: Dict[str, Any]) -> None:
        """Append a new entry. Validation is per-implementation."""
        ...

    def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve matching entries. Filter semantics per-implementation."""
        ...

    def summarize(self, max_tokens: int) -> str:
        """Produce a text summary fitting within a token budget for
        prompt injection. Implementations should be adaptive — if the
        budget is small, produce a terse summary; if large, include
        more detail."""
        ...

    def export(self) -> Dict[str, Any]:
        """Serialize for persistence or transfer."""
        ...

    def clear(self) -> None:
        """Reset the memory (e.g. on per-loop boundary)."""
        ...


class MemoryBundle:
    """Collection of memories available to an agent in a given context.

    The bundle is constructed once per nav_loop / chat_session and
    passed into `ToolContext.memory_bundle`. PromptBuilder calls
    `render_context(budgets)` to produce the memory section of the
    LLM prompt; tools can call `get("spatial")` to query memory
    directly.

    Phase 3 typically contains just one memory (SpatialMemory).
    Phase 4+ may add more, all through the same `register` call.
    """

    def __init__(self, memories: Optional[Dict[str, Memory]] = None):
        self._memories: Dict[str, Memory] = dict(memories) if memories else {}

    def register(self, memory: Memory) -> None:
        """Register a memory by its `name` attribute."""
        self._memories[memory.name] = memory

    def get(self, name: str) -> Optional[Memory]:
        """Look up a memory by name. Returns None if not registered."""
        return self._memories.get(name)

    def list_names(self) -> List[str]:
        """Return the names of all registered memories."""
        return list(self._memories.keys())

    def add_to(self, name: str, entry: Dict[str, Any]) -> None:
        """Convenience: append an entry to the named memory if it exists."""
        mem = self._memories.get(name)
        if mem is not None:
            mem.add(entry)

    def render_context(self, budgets: Dict[str, int]) -> str:
        """Render memories into a single prompt section respecting
        token budgets.

        For each (name, budget) pair in `budgets`, looks up the
        corresponding memory and calls `memory.summarize(budget)`.
        Non-empty summaries are joined with a `## {name}` header
        between them. Memories not in the bundle are silently skipped.

        Example::

            bundle.render_context({"spatial": 500, "episodic": 300})
            # → "## spatial\\nSnapshots: 5 | Rooms: ...\\n\\n## episodic\\n..."

        Returns an empty string if no memories produce output.
        """
        parts: List[str] = []
        for name, budget in budgets.items():
            mem = self._memories.get(name)
            if mem is None:
                continue
            summary = mem.summarize(budget)
            if summary:
                parts.append(f"## {name}\n{summary}")
        return "\n\n".join(parts)

    def export_all(self) -> Dict[str, Any]:
        """Serialize every memory for persistence or debugging."""
        return {name: mem.export() for name, mem in self._memories.items()}


__all__ = ["Memory", "MemoryBundle"]
