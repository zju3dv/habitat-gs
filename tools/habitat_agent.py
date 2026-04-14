#!/usr/bin/env python3
"""Legacy entry shim — unified launcher moved to ``habitat_agent.__main__``.

This file is kept so the canonical invocation
    python tools/habitat_agent.py
still works without any change. The same code is also reachable as
    python -m habitat_agent
which is the preferred form for new scripts. Will be removed at the
end of Phase 2.
"""

from __future__ import annotations

import sys

from habitat_agent.__main__ import HabitatLauncher, main

__all__ = ["HabitatLauncher", "main"]


if __name__ == "__main__":
    sys.exit(main())
