#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict


from _adapter_loader import _ensure_local_habitat_sim_package


def _load_local_adapter():
    package_dir = Path(__file__).resolve().parent.parent / "src_python" / "habitat_sim"
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


def _emit(response: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _parse_request(raw: str) -> Dict[str, Any]:
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    return request


def _json_error(action: str, message: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "action": action,
        "request_id": None,
        "session_id": None,
        "error": {"type": "ValueError", "message": message},
    }


def _run_single_request(adapter: HabitatAdapter, request_json: str) -> int:
    try:
        request = _parse_request(request_json)
    except (json.JSONDecodeError, ValueError) as exc:
        _emit(_json_error("parse_request", str(exc)))
        return 2

    _emit(adapter.handle_request(request))
    return 0


def _run_stream(adapter: HabitatAdapter) -> int:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = _parse_request(stripped)
        except (json.JSONDecodeError, ValueError) as exc:
            _emit(_json_error("parse_request", str(exc)))
            continue
        _emit(adapter.handle_request(request))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "habitat-gs bridge adapter. "
            "Reads JSON requests and writes JSON responses."
        )
    )
    parser.add_argument(
        "--request-json",
        type=str,
        default=None,
        help="One-shot JSON request string. If omitted, run in stdin stream mode.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    adapter = HabitatAdapter()
    try:
        if args.request_json is not None:
            return _run_single_request(adapter, args.request_json)
        return _run_stream(adapter)
    finally:
        adapter.close_all()


if __name__ == "__main__":
    raise SystemExit(main())
