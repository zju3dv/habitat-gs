"""Lazy loader for the openai SDK.

The SDK is imported on first use so that tools not needing LLM access
(the bridge adapter, file I/O helpers, analytics scripts) can run without
paying the import cost or requiring openai to be installed.
"""

from __future__ import annotations

import sys

_openai_module = None


def get_openai():
    """Return the imported openai module. Exits the process with a helpful
    error if openai is not installed."""
    global _openai_module
    if _openai_module is None:
        try:
            import openai as _oai
        except ImportError:
            sys.exit("ERROR: openai SDK not installed. Run: pip install openai>=1.30.0")
        _openai_module = _oai
    return _openai_module


__all__ = ["get_openai"]
