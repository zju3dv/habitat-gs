"""Cross-process file I/O primitives for habitat_agent.

This module provides:
- `acquire_nav_status_lock`   : cross-process exclusive flock on a sidecar lockfile
- `copy_file_atomic_under_lock` : snapshot a file under lock into a durable temp-rename copy
- `append_jsonl_atomic`       : durable single-line append with fsync

All three exist because `nav_status.json` has two writers (the bridge process
and the nav_agent subprocess) and earlier versions suffered from silent
data-loss races. See PR #28 state-management fixes for history.

Phase 1 note: these functions were moved here from `tools/habitat_agent_core.py`
without any behavior change. The legacy shim at `tools/habitat_agent_core.py`
re-exports them so old import paths keep working.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import time
from typing import Any, Dict, Iterator


# Sidecar lock-file extension. The lockfile lives next to the protected
# file (e.g. nav_status.json.lock) so it shares filesystem semantics with
# the file it guards. We use a sidecar instead of locking the file itself
# because the bridge writes via temp file + os.replace, which would unlink
# any flock held on the original inode.
_NAV_STATUS_LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def acquire_nav_status_lock(
    nav_status_path: str,
    *,
    timeout_s: float = 5.0,
) -> Iterator[None]:
    """Acquire a cross-process exclusive lock on a nav_status.json sidecar.

    Purpose
        Serialize concurrent read-modify-write sequences on nav_status.json
        across the bridge process and the nav_agent subprocess.

    Why this exists
        nav_status.json has two writers (bridge `_persist_json_atomic` and
        nav_agent `mark_terminal_status` fallback). Threading locks do not
        cross process boundaries; without this helper, a subprocess RMW can
        silently revert a bridge write that landed between its read and
        write phases (bug B1 in PR #28 review).

    When NOT to use
        Read-only access does not need the lock. The bridge's
        `_persist_json_atomic` uses temp-file + `os.replace`, so a reader
        always sees either the old or the new file, never a partial state.
        Acquire the lock ONLY around RMW (read-then-write) sequences.

    Failure modes
        - Raises `TimeoutError` if the lock cannot be acquired within
          `timeout_s` seconds.
        - The lock is released on context exit (even on exception) and on
          process death (kernel-managed via fcntl.flock).
    """
    lock_path = nav_status_path + _NAV_STATUS_LOCK_SUFFIX
    parent = os.path.dirname(os.path.abspath(lock_path)) or "."
    os.makedirs(parent, exist_ok=True)
    # Cross-user compatibility (Codex P2 follow-up):
    #   - Mode 0o666 so a deployment with umask 0o002 ends up with a
    #     group-writable lock file; default umask 0o022 still yields
    #     0o644 which is fine for single-user dev.
    #   - O_RDONLY is intentional: fcntl.flock on Linux is advisory and
    #     does NOT require write access on the fd. Opening with O_RDWR
    #     would raise PermissionError whenever a second process (e.g.,
    #     a nav_agent subprocess running under a different UID than the
    #     bridge) lacked write permission on an existing lock file.
    #     Read-only open works as long as the creator gave at least
    #     "other: read" (0o644 does).
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"acquire_nav_status_lock: failed to acquire {lock_path} "
                        f"within {timeout_s}s"
                    )
                time.sleep(0.02)
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def copy_file_atomic_under_lock(
    src_path: str,
    dst_path: str,
    *,
    lock_path: str,
) -> None:
    """Snapshot ``src_path`` into ``dst_path`` under a cross-process flock.

    Purpose
        Take a self-consistent backup of nav_status.json into last_good.json
        even when the bridge may be writing concurrently.

    Why this exists
        ``shutil.copy2`` reads the source in chunks then writes the
        destination, so a concurrent ``os.replace`` of the source can produce
        a torn read — the destination ends up half old, half new (bug B5).
        Using the same sidecar flock as the bridge writers serializes the
        snapshot with any in-flight write, and the destination is itself
        written via temp-file + rename so partial writes never appear.

    When NOT to use
        General-purpose copy. This helper is specific to the
        nav_status.json + last_good.json relationship and assumes both
        files are small JSON documents.

    Failure modes
        Raises ``OSError`` on read/write failure. Raises ``TimeoutError``
        if the lock cannot be acquired within 5 seconds.
    """
    parent = os.path.dirname(os.path.abspath(dst_path)) or "."
    os.makedirs(parent, exist_ok=True)
    with acquire_nav_status_lock(lock_path):
        with open(src_path, "rb") as fsrc:
            data = fsrc.read()
        tmp_path = dst_path + ".tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dst_path)


def append_jsonl_atomic(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON-line to ``path`` with ``fsync``.

    Purpose
        Durably append one record to a JSON-lines file (session_stats.jsonl,
        events.jsonl, trace.jsonl).

    Why this exists
        The previous `open(path, "a"); f.write(...)` pattern relied on user-
        space buffering. On crash/power loss, the most recent records could
        be lost (bug B4). This helper opens with `O_APPEND`, writes the line
        plus newline, calls `os.fsync` on the descriptor, and closes —
        guaranteeing that on successful return the record is persisted.

        POSIX guarantees that an `O_APPEND` write of <= PIPE_BUF bytes is
        atomic with respect to other concurrent appenders, so individual
        small JSON lines are race-free across processes.

    When NOT to use
        Bulk writes or non-jsonl payloads. This helper assumes one record =
        one JSON line and is intentionally minimal.

    Failure modes
        Raises `OSError` on disk-full, permission-denied, EBADF, etc. The
        caller MUST handle the failure (typically: log + raise) — silently
        swallowing the error reintroduces B4.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Open with O_APPEND so the kernel handles concurrent appenders;
    # avoid Python's high-level open("a") so we can fsync the raw fd.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        data = line.encode("utf-8")
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if n <= 0:
                raise OSError(errno.EIO, f"append_jsonl_atomic: short write to {path}")
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "_NAV_STATUS_LOCK_SUFFIX",
    "acquire_nav_status_lock",
    "copy_file_atomic_under_lock",
    "append_jsonl_atomic",
]
