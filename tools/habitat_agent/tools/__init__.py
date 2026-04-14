"""habitat_agent.tools — Tool abstraction layer + concrete Tool subclasses.

Phase 2 replaced the legacy `build_tool_schemas` / `ToolExecutor`
pair with a declarative `Tool` / `ToolRegistry` system in `base.py`.
There are 16 concrete Tool subclasses organized by category:

- `navigation.py`  — forward, turn, navigate, find_path, sample_point
- `perception.py`  — look, panorama, depth_analyze
- `mapping.py`     — topdown
- `status.py`      — update_nav_status, export_video
- `session.py`     — init_scene, close_session, nav_loop_start,
                     nav_loop_status, nav_loop_stop (chat-only)

Importing `habitat_agent.tools` triggers registration of every Tool
subclass with `ToolRegistry`. This is the *only* place where the
registration order is guaranteed — callers that need to be certain
every tool is present (e.g. `mcp_server` building its hab_* list, or
`nav_agent` calling `ToolRegistry.build_openai_schemas`) should import
this package first, then query the registry.

For how to add a new tool, see `docs/tool-authoring-guide.md`.
"""

from . import base  # noqa: F401  — re-exported as habitat_agent.tools.base

# Importing each Tool category module triggers `ToolRegistry.register`
# at module-import time, which is why the explicit imports below matter.
# Do not convert these into `import *` or a star re-export — the order
# and the side effect are the point.
from . import navigation  # noqa: F401
from . import perception  # noqa: F401
from . import mapping  # noqa: F401
from . import status  # noqa: F401
from . import session  # noqa: F401
