"""Shared helper: ensure local src_python/habitat_sim package is importable."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _ensure_local_habitat_sim_package(package_dir: Path) -> None:
    package_name = "habitat_sim"
    local_path = str(package_dir)

    existing = sys.modules.get(package_name)
    if existing is None:
        try:
            importlib.import_module(package_name)
            existing = sys.modules.get(package_name)
        except Exception:  # noqa: BLE001
            existing = None

    if existing is None:
        package = types.ModuleType(package_name)
        package.__path__ = [local_path]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
        return

    package_paths = getattr(existing, "__path__", None)
    if package_paths is None:
        package = types.ModuleType(package_name)
        package.__dict__.update(existing.__dict__)
        package.__path__ = [local_path]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
        return

    if local_path not in package_paths:
        package_paths.insert(0, local_path)
