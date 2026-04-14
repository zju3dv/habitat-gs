"""HTTP client for the habitat bridge (POST /v1/request).

`BridgeClient` is used by:
- `nav_agent` subprocess to execute tool actions
- `chat_agent` to drive the simulator from TUI chat
- `mcp_server` to forward MCP tool calls
- TUI dashboard to poll status

Moved here from `tools/habitat_agent_core.py` in Phase 1 with no behavior
change. The legacy shim at that path re-exports this class.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class BridgeClient:
    """Minimal HTTP client for the habitat bridge (POST /v1/request)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18911):
        self.base_url = f"http://{host}:{port}"
        self._request_counter = 0
        self.session_id: Optional[str] = None

    def _next_id(self) -> str:
        self._request_counter += 1
        return f"hab-{self._request_counter}"

    def healthz(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def call(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Issue a single POST to the bridge HTTP API.

        ``timeout`` — optional per-call override (seconds). Defaults to 60s
        for long-running actions. Shutdown-critical callers
        (mark_terminal_status, signal handlers) should pass a short value
        (~3s) so a hung bridge handler cannot block process termination
        for a full minute.
        """
        envelope: Dict[str, Any] = {
            "request_id": self._next_id(),
            "action": action,
            "payload": payload or {},
        }
        if self.session_id:
            envelope["session_id"] = self.session_id
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/request",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        effective_timeout = 60 if timeout is None else float(timeout)
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"Bridge HTTP {e.code}: {error_body}") from e
        if not result.get("ok", True):
            err = result.get("error", {})
            raise RuntimeError(f"Bridge error: {err.get('message', result)}")
        return result.get("result", result)


__all__ = ["BridgeClient"]
