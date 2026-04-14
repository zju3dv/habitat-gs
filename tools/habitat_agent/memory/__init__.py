"""habitat_agent.memory — pluggable memory abstraction layer.

Provides the `Memory` protocol and `MemoryBundle` for uniform access
to agent memories. Currently contains one concrete implementation
(`SpatialMemory`), with episodic / procedural / reference memory types
available for future extension.

Usage::

    from habitat_agent.memory.base import MemoryBundle
    from habitat_agent.memory.spatial import SpatialMemory

    bundle = MemoryBundle()
    bundle.register(SpatialMemory("/path/to/spatial_memory.json"))
    prompt_section = bundle.render_context({"spatial": 500})
"""

from . import base  # noqa: F401
from . import spatial  # noqa: F401
