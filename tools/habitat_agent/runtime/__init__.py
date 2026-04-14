"""habitat_agent.runtime — low-level infrastructure for the agent runtime.

Modules here have no dependencies on agents or interfaces. They expose:
- `bridge_client` : HTTP client for the habitat bridge
- `file_io`       : cross-process locks, atomic write/append, copy
- `image_io`      : image encoding helpers (base64, data URL, content parts)
- `llm_client`    : lazy OpenAI SDK loader
- `config`        : dotenv loading
- `constants`     : shared constants (TERMINAL_STATUSES, etc.)

Phase 2+ will add:
- `bridge_process`   : subprocess lifecycle for bridge (Phase 1 PR 4)
- `terminal_status`  : mark_terminal_status (Phase 1 PR 2)
"""

from .constants import TERMINAL_STATUSES

__all__ = ["TERMINAL_STATUSES"]
