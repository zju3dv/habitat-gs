"""Terminal status writing for nav_agent subprocess.

This module hosts the two-tier terminal-status write strategy:

1. **Path 1 — bridge RPC** (preferred, single-owner):
   The nav_agent subprocess sends an ``update_nav_loop_status`` patch
   to the bridge via RPC with a short 3s timeout. The bridge merges
   the patch under its own lock and persists atomically. No
   cross-process race possible.

2. **Path 2 — locked local write** (fallback):
   When the bridge is unreachable or its RPC raises, the subprocess
   falls back to a direct file write guarded by a cross-process flock
   on the sidecar lock file. Uses temp-file + os.replace + fsync so
   the terminal state survives crashes.

See PR #28 state-management history for the bug chain that led to
this design (B1/B2 cross-process race, B3 fsync, the Codex follow-up
on short timeout for shutdown paths).

Moved verbatim from ``tools/nav_agent.py`` in Phase 1 PR 2 — no
behaviour changes. The old nav_agent.py re-exports these symbols for
backward compatibility until Phase 2.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .file_io import acquire_nav_status_lock
from .log import log

if TYPE_CHECKING:
    from .bridge_client import BridgeClient


# Short timeout for the bridge path of mark_terminal_status. Tuned for
# shutdown paths where a full 60s HTTP timeout would cause Ctrl+C to look
# wedged (and miss container terminationGracePeriod windows). 3 seconds
# is long enough for a healthy bridge to respond and short enough to
# fall back to the local locked write before any orchestrator loses
# patience (Codex P2 follow-up).
_TERMINAL_STATUS_BRIDGE_TIMEOUT_S = 3.0


def _write_terminal_status_locally(
    status_file: str, terminal_status: str, error_msg: str
) -> None:
    """Fallback writer for nav_status terminal state.

    Acquires the cross-process flock on the nav_status sidecar so the
    write cannot race with concurrent bridge writes (bug B1). Writes via
    temp-file + os.replace + fsync (bug B2).

    Used ONLY by mark_terminal_status when the bridge RPC path fails or
    when no bridge is provided. The lock guarantees that even if the
    bridge process briefly recovers between our read and write, no
    concurrent bridge writer can land in the middle.
    """
    with acquire_nav_status_lock(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
            nav = json.load(f)
        nav["status"] = terminal_status
        nav["error"] = error_msg
        nav["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nav["state_version"] = (nav.get("state_version") or 0) + 1

        # Atomic write with fsync — same pattern as the bridge's
        # _persist_json_atomic, kept locally to avoid coupling nav_agent
        # to the bridge module.
        tmp_path = status_file + ".tmp"
        parent = os.path.dirname(os.path.abspath(status_file)) or "."
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            data = json.dumps(nav, ensure_ascii=False, indent=2).encode("utf-8")
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, status_file)
        # fsync the parent directory so the rename is durable
        try:
            dir_fd = os.open(parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (AttributeError, OSError):
            # Best-effort: some filesystems / OSes don't support O_DIRECTORY
            pass


def mark_terminal_status(
    status_file: str,
    terminal_status: str,
    error_msg: str,
    *,
    bridge: Optional["BridgeClient"] = None,
    loop_id: Optional[str] = None,
) -> None:
    """Set the nav-loop terminal status (status + error fields).

    Two-tier write strategy:

    Path 1 — bridge RPC (preferred, single-owner):
        When ``bridge`` and ``loop_id`` are provided, send the patch via
        ``bridge.call("update_nav_loop_status", ..., timeout=3.0)``. A
        short timeout is mandatory because this function is called from
        shutdown paths (SIGTERM, max iterations, fatal exception) where
        a hung bridge handler must not block termination.
        The bridge merges the patch under its own ``_nav_loop_lock`` and
        persists atomically, keeping ``record.nav_status`` and the file
        consistent. No cross-process race possible.

    Path 2 — locked local write (fallback, B1 + B2 fix):
        When the bridge is None, the RPC raises, or the short timeout
        fires, write the file directly. The write is guarded by a
        cross-process flock on ``<status_file>.lock`` so a recovering
        bridge cannot race with us, and it uses temp-file + os.replace
        + fsync so the change survives crashes.

    The function never raises — it logs failures and returns. This matches
    the original contract: terminal-status setting is best-effort because
    by the time we get here the agent is already exiting; we want to
    avoid masking the original failure.
    """
    # Path 1: bridge first, with short timeout
    if bridge is not None and loop_id:
        try:
            bridge.call(
                "update_nav_loop_status",
                {
                    "loop_id": loop_id,
                    "patch": {
                        "status": terminal_status,
                        "error": error_msg,
                    },
                },
                timeout=_TERMINAL_STATUS_BRIDGE_TIMEOUT_S,
            )
            return
        except Exception as exc:
            log(
                f"WARNING: mark_terminal_status: bridge RPC failed "
                f"({exc!r}); falling back to local locked write"
            )

    # Path 2: locked local write
    try:
        _write_terminal_status_locally(status_file, terminal_status, error_msg)
    except Exception as exc:
        log(f"ERROR: mark_terminal_status: local fallback failed: {exc!r}")


__all__ = [
    "_TERMINAL_STATUS_BRIDGE_TIMEOUT_S",
    "_write_terminal_status_locally",
    "mark_terminal_status",
]
