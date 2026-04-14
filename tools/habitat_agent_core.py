"""Legacy shim — code moved to `habitat_agent/` package in Phase 1.

This file is kept so existing imports like
    from habitat_agent_core import BridgeClient, acquire_nav_status_lock, ...
continue to work without modification.

Phase 2 removed the `ToolExecutor` / `build_tool_schemas` legacy
pair (PR 5) because the dynamic `ToolRegistry` dispatch path in
`habitat_agent.tools.base` fully replaced them. New code should
import tools through the Registry:

    import habitat_agent.tools                 # triggers registration
    from habitat_agent.tools.base import ToolRegistry, ToolContext

The runtime primitives below are still re-exported because they have
no replacement — they're stable infrastructure that other packages
rely on.

This shim itself will be removed once every external caller has been
migrated away from the `habitat_agent_core` namespace.
"""

from __future__ import annotations

# Runtime primitives (PR #28 file I/O, HTTP client, LLM loader, config)
from habitat_agent.runtime.bridge_client import BridgeClient
from habitat_agent.runtime.config import load_dotenv_from_project
from habitat_agent.runtime.constants import TERMINAL_STATUSES
from habitat_agent.runtime.file_io import (
    _NAV_STATUS_LOCK_SUFFIX,
    acquire_nav_status_lock,
    append_jsonl_atomic,
    copy_file_atomic_under_lock,
)
from habitat_agent.runtime.image_io import (
    build_image_content_parts,
    read_image_as_data_url,
    read_image_b64,
)
from habitat_agent.runtime.llm_client import get_openai

__all__ = [
    "BridgeClient",
    "load_dotenv_from_project",
    "TERMINAL_STATUSES",
    "_NAV_STATUS_LOCK_SUFFIX",
    "acquire_nav_status_lock",
    "append_jsonl_atomic",
    "copy_file_atomic_under_lock",
    "build_image_content_parts",
    "read_image_as_data_url",
    "read_image_b64",
    "get_openai",
]
