#!/usr/bin/env python3
"""HabitatAgent unified launcher — start bridge, MCP, TUI, and dashboard
in one command based on the desired use case.

Usage:
    python3 tools/habitat_agent.py              # TUI + bridge (default)
    python3 tools/habitat_agent.py --mcp        # TUI + bridge + MCP server
    python3 tools/habitat_agent.py --all        # TUI + bridge + MCP + dashboard
    python3 tools/habitat_agent.py --headless   # bridge + MCP (no TUI)
    python3 tools/habitat_agent.py --bridge-only  # bridge only (for CLI debugging)
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1]  # legacy tools/ dir


def _subprocess_env(
    extra: dict[str, str] | None = None,
    *,
    isolate_python: bool = False,
) -> dict[str, str]:
    """Build child-process environment, with optional user-site isolation."""
    env = os.environ.copy()
    if isolate_python:
        env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update(extra)
    return env


def _tail_text_file(path: str, lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            chunks = f.readlines()
    except Exception:
        return ""
    return "".join(chunks[-lines:]).rstrip()


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        env_path = _TOOLS_DIR.parent / ".env"
        if env_path.is_file():
            load_dotenv(str(env_path), override=False)
    except ImportError:
        pass


def _python_bin():
    return sys.executable


def _check_health(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_health(host: str, port: int, retries: int = 20, interval: float = 0.5) -> bool:
    for _ in range(retries):
        if _check_health(host, port):
            return True
        time.sleep(interval)
    return False


class HabitatLauncher:
    """Manages lifecycle of bridge, MCP server, dashboard, and TUI."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bridge_proc: subprocess.Popen | None = None
        self.mcp_proc: subprocess.Popen | None = None
        self.dashboard_proc: subprocess.Popen | None = None
        self.tui_proc: subprocess.Popen | None = None
        self._bridge_owned = False  # True if we started bridge (vs. pre-existing)
        self._shutting_down = False

    def start_bridge(self) -> bool:
        host = self.args.host
        port = self.args.port

        if _check_health(host, port):
            print(f"[habitat-agent] Bridge already running at {host}:{port}")
            self._bridge_owned = False
            return True

        bridge_script = str(_TOOLS_DIR / "habitat_agent_server.py")
        bridge_log = self.args.bridge_log
        os.makedirs(os.path.dirname(bridge_log) or ".", exist_ok=True)

        env = _subprocess_env({"HABITAT_SIM_LOG": "quiet", "MAGNUM_LOG": "QUIET"})

        cmd = [
            _python_bin(), bridge_script,
            "--host", host,
            "--port", str(port),
            "--log-format", "text",
            "--session-idle-timeout-s", "900",
        ]
        log_fh = open(bridge_log, "a", encoding="utf-8")
        self.bridge_proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            env=env, close_fds=True,
        )
        log_fh.close()
        print(f"[habitat-agent] Bridge starting (pid={self.bridge_proc.pid}, log={bridge_log})")

        if _wait_for_health(host, port):
            print(f"[habitat-agent] Bridge ready at {host}:{port}")
            self._bridge_owned = True
            return True
        else:
            print(f"[habitat-agent] Bridge failed to start. Check {bridge_log}", file=sys.stderr)
            return False

    def start_mcp(self) -> bool:
        mcp_script = str(_TOOLS_DIR / "mcp_server.py")
        mcp_port = self.args.mcp_port
        mcp_log = self.args.mcp_log
        os.makedirs(os.path.dirname(os.path.abspath(mcp_log)), exist_ok=True)

        cmd = [
            _python_bin(), mcp_script,
            "--transport", "streamable-http",
            "--port", str(mcp_port),
        ]
        log_fh = open(mcp_log, "a", encoding="utf-8")
        self.mcp_proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            env=_subprocess_env(isolate_python=True),
            close_fds=True,
        )
        log_fh.close()
        print(f"[habitat-agent] MCP server starting (pid={self.mcp_proc.pid}, port={mcp_port}, log={mcp_log})")
        # MCP server doesn't have a health endpoint, give it a moment
        time.sleep(1.0)
        if self.mcp_proc.poll() is not None:
            tail = _tail_text_file(mcp_log)
            print(f"[habitat-agent] MCP server exited immediately. Check {mcp_log}", file=sys.stderr)
            if tail:
                print(f"[habitat-agent] Last log lines:\n{tail}", file=sys.stderr)
            return False
        print(f"[habitat-agent] MCP server ready at http://127.0.0.1:{mcp_port}/mcp")
        return True

    def start_dashboard(self) -> bool:
        dashboard_script = str(_TOOLS_DIR / "analytics" / "nav_dashboard.py")
        if not os.path.isfile(dashboard_script):
            print("[habitat-agent] Dashboard script not found, skipping", file=sys.stderr)
            return False

        dash_port = self.args.dashboard_port
        dash_log = self.args.dashboard_log
        os.makedirs(os.path.dirname(os.path.abspath(dash_log)), exist_ok=True)

        cmd = [_python_bin(), dashboard_script, "--port", str(dash_port)]
        log_fh = open(dash_log, "a", encoding="utf-8")
        self.dashboard_proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            env=_subprocess_env(),
            close_fds=True,
        )
        log_fh.close()
        print(f"[habitat-agent] Dashboard starting (pid={self.dashboard_proc.pid}, port={dash_port})")
        return True

    def start_tui(self) -> int:
        tui_script = str(_TOOLS_DIR / "habitat_agent_tui.py")
        cmd = [
            _python_bin(), tui_script,
            "--host", self.args.host,
            "--port", str(self.args.port),
            "--no-start-bridge",  # bridge already managed by us
            "--keep-bridge",      # don't let TUI kill our bridge
        ]
        self.tui_proc = subprocess.Popen(cmd)
        self.tui_proc.wait()
        return self.tui_proc.returncode or 0

    def run(self) -> int:
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Always start bridge first
        if not self.start_bridge():
            self.shutdown()
            return 1

        # Start optional services
        want_mcp = self.args.mcp or self.args.all or self.args.headless
        want_dashboard = self.args.dashboard or self.args.all
        want_tui = not self.args.headless and not self.args.bridge_only

        if want_mcp:
            if not self.start_mcp():
                self.shutdown()
                return 1

        if want_dashboard:
            self.start_dashboard()

        if want_tui:
            # TUI runs in foreground (blocking)
            rc = self.start_tui()
            self.shutdown()
            return rc
        else:
            # Headless / bridge-only: wait for SIGINT
            services = ["bridge"]
            if want_mcp:
                services.append("MCP")
            print(f"[habitat-agent] Running headless ({', '.join(services)}). Press Ctrl+C to stop.")
            try:
                while True:
                    # Check if subprocesses are still alive
                    if self.bridge_proc and self.bridge_proc.poll() is not None:
                        print("[habitat-agent] Bridge exited unexpectedly", file=sys.stderr)
                        break
                    if want_mcp and self.mcp_proc and self.mcp_proc.poll() is not None:
                        print("[habitat-agent] MCP server exited unexpectedly", file=sys.stderr)
                        break
                    time.sleep(2.0)
            except KeyboardInterrupt:
                pass
            self.shutdown()
            return 0

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        print("\n[habitat-agent] Shutting down...")

        for name, proc in [
            ("TUI", self.tui_proc),
            ("Dashboard", self.dashboard_proc),
            ("MCP", self.mcp_proc),
            ("Bridge", self.bridge_proc),
        ]:
            if proc is None or proc.poll() is not None:
                continue
            print(f"[habitat-agent] Stopping {name} (pid={proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        # Only kill port holders for services we started ourselves
        if self._bridge_owned:
            self._kill_port_holder(self.args.port, "Bridge")
        if (self.args.mcp or self.args.all or self.args.headless) and self.mcp_proc is not None:
            self._kill_port_holder(self.args.mcp_port, "MCP")

        print("[habitat-agent] All services stopped.")

    def _kill_port_holder(self, port: int, label: str):
        """Find and kill any process listening on the given port."""
        try:
            out = subprocess.check_output(
                ["ss", "-tlnp", f"sport = :{port}"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                # Extract pid from e.g. pid=12345
                m = re.search(r"pid=(\d+)", line)
                if m:
                    pid = int(m.group(1))
                    # Don't kill ourselves
                    if pid == os.getpid():
                        continue
                    print(f"[habitat-agent] Killing external {label} (pid={pid}, port={port})")
                    os.kill(pid, signal.SIGTERM)
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass

    def _signal_handler(self, signum, frame):
        self.shutdown()
        raise SystemExit(0)


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="HabitatAgent — unified launcher for navigation agent services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                    Interactive TUI + bridge (default)
  %(prog)s --mcp              TUI + bridge + MCP server
  %(prog)s --all              TUI + bridge + MCP + dashboard
  %(prog)s --headless         Bridge + MCP (no TUI, for CI/headless deployment)
  %(prog)s --bridge-only      Bridge only (for hab CLI debugging)
""",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bridge host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18911, help="Bridge port (default: 18911)")
    parser.add_argument("--mcp", action="store_true", help="Also start MCP server")
    parser.add_argument("--mcp-port", type=int, default=18912, help="MCP server port (default: 18912)")
    parser.add_argument("--dashboard", action="store_true", help="Also start web dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--all", action="store_true", help="Start all services (TUI + bridge + MCP + dashboard)")
    parser.add_argument("--headless", action="store_true", help="Bridge + MCP without TUI (for CI/deployment)")
    parser.add_argument("--bridge-only", action="store_true", help="Only start bridge (for CLI debugging)")
    _u = os.getenv("USER", "user")
    parser.add_argument("--bridge-log", default=f"/tmp/habitat_agent_bridge_{_u}.log", help="Bridge log file")
    parser.add_argument("--mcp-log", default=f"/tmp/habitat_agent_mcp_{_u}.log", help="MCP server log file")
    parser.add_argument("--dashboard-log", default=f"/tmp/habitat_agent_dashboard_{_u}.log", help="Dashboard log file")

    args = parser.parse_args()

    launcher = HabitatLauncher(args)
    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
