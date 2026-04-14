#!/usr/bin/env python3
"""Legacy entry shim — code moved to habitat_agent.interfaces.mcp_server.

This file is kept so the canonical invocation
    python tools/mcp_server.py
still works without any change. It also re-exports the FastMCP
``mcp`` instance and every ``hab_*`` tool so backward-compat holds
until **end of Phase 2**.
"""

from __future__ import annotations

import sys

from habitat_agent.interfaces.mcp_server import main, mcp  # noqa: F401

# Re-export every hab_* tool function so callers that did
# `from mcp_server import hab_init` keep working.
from habitat_agent.interfaces import mcp_server as _impl
for _name in dir(_impl):
    if _name.startswith("hab_") or _name in {"_bridge", "_artifacts_dir"}:
        globals()[_name] = getattr(_impl, _name)
del _name, _impl

__all__ = ["mcp", "main"]


if __name__ == "__main__":
    main()
    sys.exit(0)
