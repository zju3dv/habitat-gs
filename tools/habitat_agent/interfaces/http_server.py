#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import importlib.util
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple


from _adapter_loader import _ensure_local_habitat_sim_package


def _load_local_adapter():
    package_dir = Path(__file__).resolve().parents[3] / "src_python" / "habitat_sim"
    module_path = package_dir / "habitat_adapter.py"
    _ensure_local_habitat_sim_package(package_dir)
    spec = importlib.util.spec_from_file_location(
        "habitat_sim.habitat_adapter", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load HabitatAdapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HabitatAdapter


HabitatAdapter = _load_local_adapter()


LOGGER = logging.getLogger("habitat_agent.http")


class _ReusableHTTPServer(HTTPServer):
    # Habitat-Sim OpenGL context is thread-affine; keep all requests on one thread.
    allow_reuse_address = True


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, default=_json_default).encode(
        "utf-8"
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return list(value)
        except Exception:  # noqa: BLE001
            pass
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=True)


def _read_request_json(handler: BaseHTTPRequestHandler) -> Tuple[Dict[str, Any], int]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError:
        return {"ok": False, "error": "Invalid Content-Length"}, 400

    body = handler.rfile.read(max(length, 0))
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON body: {exc}"}, 400

    if not isinstance(payload, dict):
        return {"ok": False, "error": "JSON body must be an object"}, 400
    return payload, 200


def build_handler(
    adapter: HabitatAdapter,
    logger: logging.Logger,
    access_log: bool,
    session_idle_timeout_s: float,
):
    class _Handler(BaseHTTPRequestHandler):
        def _reap_idle_sessions(self) -> None:
            expired_session_ids = adapter.reap_idle_sessions(session_idle_timeout_s)
            if expired_session_ids:
                logger.info(
                    "Reaped idle sessions count=%d ids=%s",
                    len(expired_session_ids),
                    ",".join(expired_session_ids),
                )

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            try:
                data = _json_bytes(payload)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except TypeError as exc:
                logger.exception("Failed to serialize response payload: %s", exc)
                fallback = _json_bytes(
                    {
                        "ok": False,
                        "error": "Response serialization failed",
                        "details": str(exc),
                    }
                )
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(fallback)))
                self.end_headers()
                self.wfile.write(fallback)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Client disconnected before response was sent.")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._reap_idle_sessions()
                runtime = adapter.get_runtime_status()
                self._send(
                    200,
                    {
                        "ok": True,
                        "status": "live",
                        "service": "habitat-gs-http-server",
                        **runtime,
                    },
                )
                if access_log:
                    logger.info(
                        "GET /healthz -> 200 active_sessions=%s uptime_s=%s",
                        runtime["active_sessions"],
                        runtime["uptime_s"],
                    )
                return
            self._send(404, {"ok": False, "error": "Not Found"})
            if access_log:
                logger.info("GET %s -> 404", self.path)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/request":
                self._send(404, {"ok": False, "error": "Not Found"})
                if access_log:
                    logger.info("POST %s -> 404", self.path)
                return

            self._reap_idle_sessions()
            request, code = _read_request_json(self)
            if code != 200:
                self._send(code, request)
                if access_log:
                    logger.info("POST /v1/request -> %d (invalid request body)", code)
                return

            start_time = time.perf_counter()
            response = adapter.handle_request(request)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            self._send(200, response)
            if access_log:
                logger.info(
                    "POST /v1/request action=%s request_id=%s session_id=%s ok=%s elapsed_ms=%s",
                    request.get("action"),
                    request.get("request_id"),
                    request.get("session_id"),
                    response.get("ok"),
                    elapsed_ms,
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args
            return

    return _Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HabitatAgent HTTP bridge server.\n"
            "POST JSON to /v1/request and GET /healthz for liveness."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=18911, type=int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str,
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Log per-request access lines.",
    )
    parser.add_argument(
        "--log-format",
        default="text",
        choices=["text", "json"],
        type=str,
        help="Logging format for bridge lifecycle and access lines.",
    )
    parser.add_argument(
        "--session-idle-timeout-s",
        default=0.0,
        type=float,
        help="Close sessions idle for longer than this many seconds. 0 disables reaping.",
    )
    return parser


def _configure_logging(log_level: str, log_format: str) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)


def main() -> int:
    # Auto-load .env from project root if present
    try:
        from dotenv import load_dotenv

        # http_server.py lives at tools/habitat_agent/interfaces/http_server.py,
        # so three "..": interfaces -> habitat_agent -> tools -> <repo root>.
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env")
        if os.path.isfile(_env_path):
            load_dotenv(_env_path, override=False)
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args()
    if os.environ.get("HABITAT_SIM_LOG") is None:
        os.environ["HABITAT_SIM_LOG"] = "quiet"
    if os.environ.get("MAGNUM_LOG") is None:
        os.environ["MAGNUM_LOG"] = "QUIET"
    _configure_logging(args.log_level, args.log_format)

    adapter = HabitatAdapter(logger=LOGGER)
    # Tell adapter its own listen address so nav_agent subprocesses can connect back
    adapter.bridge_host = args.host
    adapter.bridge_port = args.port
    handler = build_handler(
        adapter,
        LOGGER,
        args.access_log,
        session_idle_timeout_s=max(args.session_idle_timeout_s, 0.0),
    )
    try:
        server = _ReusableHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        LOGGER.error(
            "Failed to bind %s:%d (%s). Check if the port is already in use.",
            args.host,
            args.port,
            exc,
        )
        return 1

    stop_requested = threading.Event()

    def _request_shutdown(reason: str) -> None:
        if stop_requested.is_set():
            return
        stop_requested.set()
        LOGGER.info("Shutdown requested (%s).", reason)
        threading.Thread(target=server.shutdown, daemon=True).start()

    def _signal_handler(signum: int, _frame: Any) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:  # noqa: BLE001
            signame = f"signal:{signum}"
        _request_shutdown(signame)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    LOGGER.info(
        "Bridge started at http://%s:%d (endpoints: GET /healthz, POST /v1/request, log_format=%s, idle_timeout_s=%s)",
        args.host,
        args.port,
        args.log_format,
        max(args.session_idle_timeout_s, 0.0),
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        _request_shutdown("KeyboardInterrupt")
    finally:
        LOGGER.info("Closing active sessions...")
        adapter.close_all()
        server.server_close()
        LOGGER.info("Bridge stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
