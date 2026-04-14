"""Configuration loading helpers.

Currently only wraps python-dotenv for project-root `.env` discovery.
Kept as a separate module so the dependency on `dotenv` stays optional.
"""

from __future__ import annotations

from pathlib import Path


def load_dotenv_from_project() -> None:
    """Auto-load .env from project root if present.

    No-op if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return

    # Walk up from this file to the project root (tools/habitat_agent/runtime → tools → repo root)
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.is_file():
        _load(str(env_path), override=False)


__all__ = ["load_dotenv_from_project"]
