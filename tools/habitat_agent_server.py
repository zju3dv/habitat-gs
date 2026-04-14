#!/usr/bin/env python3
"""Legacy entry shim — bridge HTTP server moved to
``habitat_agent.interfaces.http_server``.

This file is kept so the canonical invocation
    python tools/habitat_agent_server.py
still works without any change. It also re-exports the public symbols
that other tools (the TUI auto-bridge starter, the unified launcher)
import. Will be removed at the end of Phase 2.
"""

from __future__ import annotations

import sys

from habitat_agent.interfaces.http_server import (
    build_handler,
    build_parser,
    main,
)

__all__ = ["build_handler", "build_parser", "main"]


if __name__ == "__main__":
    sys.exit(main())
