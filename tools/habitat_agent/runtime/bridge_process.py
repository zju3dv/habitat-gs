"""Bridge subprocess lifecycle for the TUI dashboard.

`BridgeProcess` starts and supervises a habitat_agent_server.py
subprocess for the TUI dashboard's "auto-bridge" mode. It is
intentionally minimal — start / stop / restart / status_text — and
can also wrap an externally-managed bridge (e.g. one started by the
unified `habitat_agent.py` launcher), in which case `process` stays
None and `status_text()` checks the bridge healthz endpoint.

Moved verbatim from `tools/habitat_agent_tui.py` in Phase 1 PR 4. No
behaviour changes.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BridgeProcess:
    host: str
    port: int
    python_bin: str
    bridge_script: str
    log_path: str
    log_format: str
    session_idle_timeout_s: float
    access_log: bool
    process: Optional[subprocess.Popen[Any]] = None
    started_by_tui: bool = False
    last_error: Optional[str] = None

    def start(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return True
        cmd = [
            self.python_bin,
            self.bridge_script,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-format",
            self.log_format,
            "--session-idle-timeout-s",
            str(max(self.session_idle_timeout_s, 0.0)),
        ]
        if self.access_log:
            cmd.append("--access-log")

        env = os.environ.copy()
        env.setdefault("HABITAT_SIM_LOG", "quiet")
        env.setdefault("MAGNUM_LOG", "QUIET")
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        try:
            log_fh = open(self.log_path, "a", encoding="utf-8")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    close_fds=True,
                )
            finally:
                # Parent process should close its own fd copy immediately.
                log_fh.close()
            self.started_by_tui = True
            self.last_error = None
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self.process = None
            return False

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self.process = None

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def status_text(self) -> str:
        if self.process is None:
            # Bridge managed externally (e.g. by habitat_agent.py launcher)
            # Check health to show meaningful status
            try:
                req = urllib.request.Request(
                    f"http://{self.host}:{self.port}/healthz", method="GET"
                )
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        return "running(external)"
            except Exception:
                pass
            return "not running"
        rc = self.process.poll()
        if rc is None:
            return f"running(pid={self.process.pid})"
        return f"exited({rc})"


__all__ = ["BridgeProcess"]
