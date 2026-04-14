from __future__ import annotations

import contextlib
import copy
import datetime
import fcntl
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

import numpy as np

# Sidecar lock-file suffix. Kept identical to the subprocess-side helper in
# tools/habitat_agent_core.py so both sides lock the same path. The two
# implementations are intentional mirrors: the bridge cannot import from
# tools/ (not on the habitat_sim package path), and tools/ should not
# depend on the installed bridge adapter.
_NAV_STATUS_LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def _acquire_nav_status_lock(
    nav_status_path: str,
    *,
    timeout_s: float = 5.0,
) -> Iterator[None]:
    """Bridge-side mirror of tools/habitat_agent_core.acquire_nav_status_lock.

    Acquires a cross-process exclusive lock on a sidecar ``<path>.lock``
    file using fcntl.flock. Both the bridge (this module) and the
    nav_agent subprocess (tools/habitat_agent_core.py) call this helper
    with the same ``nav_status_path`` so their writes serialize.

    Why two copies exist: the installed habitat_sim bridge package cannot
    import from the repo's tools/ directory, and tools/ should not depend
    on the adapter package — so the helper is duplicated. If you change
    the lock protocol on one side, change it on the other.
    """
    lock_path = nav_status_path + _NAV_STATUS_LOCK_SUFFIX
    parent = os.path.dirname(os.path.abspath(lock_path)) or "."
    os.makedirs(parent, exist_ok=True)
    # Cross-user compatibility: see the matching rationale block in
    # tools/habitat_agent_core.py:acquire_nav_status_lock. The two
    # helpers must use the exact same open flags and mode so bridge
    # and subprocess can interoperate on the same sidecar file.
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
                        f"_acquire_nav_status_lock: failed to acquire "
                        f"{lock_path} within {timeout_s}s"
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

from .types import (
    _MAX_NAV_ACTION_HISTORY,
    _NAV_LOOP_SCRIPT,
    _NAV_STATUS_ALLOWED_VALUES,
    _NAV_STATUS_IMMUTABLE_FIELDS,
    _NAV_STATUS_PATCH_FIELDS,
    HabitatAdapterError,
)


class HabitatAdapterPatchMixin:
    """Nav status patch validation, normalization, and coercion helpers."""

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    @staticmethod
    def _coerce_non_negative_int(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HabitatAdapterError(f'Field "{field_name}" must be an int >= 0')
        return int(value)

    @staticmethod
    def _coerce_optional_float(value: Any, field_name: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise HabitatAdapterError(
                f'Field "{field_name}" must be a float or null'
            )
        if not isinstance(value, (int, float)):
            raise HabitatAdapterError(
                f'Field "{field_name}" must be a float or null'
            )
        return float(value)

    @staticmethod
    def _persist_nav_status_locked(path: str, payload: Mapping[str, Any]) -> None:
        """Persist a nav_status.json payload under the cross-process flock.

        ALL bridge-side writes to nav_status.json files must go through
        this wrapper (not the raw _persist_json_atomic) so that concurrent
        subprocess fallback writes via mark_terminal_status cannot race.

        The Codex follow-up of the PR #28 state-management fixes identified
        that _get_nav_loop_status's refresh-and-persist path was bypassing
        the sidecar lock the subprocess side acquired, producing the same
        class of cross-process race the lock was meant to prevent.

        Writes to non-nav_status files (e.g. spatial_memory.json) continue
        to use _persist_json_atomic directly — they don't participate in
        the subprocess-side fallback protocol.
        """
        # Call module-level function so tests can monkeypatch it.
        with _acquire_nav_status_lock(path):
            HabitatAdapterPatchMixin._persist_json_atomic(path, payload)

    @staticmethod
    def _persist_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(abs_path)}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, abs_path)
            try:
                dir_fd = os.open(parent, os.O_DIRECTORY)
            except (AttributeError, OSError):
                return
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    _SPATIAL_MEMORY_MAX_SNAPSHOTS = 500

    def _apply_spatial_memory_append(
        self, entries: Any, spatial_memory_file: str
    ) -> None:
        """Validate and append observation entries to the spatial memory file."""
        if not isinstance(entries, list):
            raise HabitatAdapterError(
                '"spatial_memory_append" must be a list of observation objects'
            )
        if not spatial_memory_file or not isinstance(spatial_memory_file, str):
            raise HabitatAdapterError(
                '"spatial_memory_append" requires a valid spatial_memory_file in nav_status'
            )
        if not entries:
            return

        validated: list[dict] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise HabitatAdapterError(
                    f'"spatial_memory_append[{i}]" must be an object'
                )
            obs: dict = {}
            if "position" in entry:
                pos = entry["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    obs["position"] = [float(x) for x in pos[:3]]
            if "heading_deg" in entry:
                obs["heading_deg"] = float(entry["heading_deg"])
            for str_field in ("scene_description", "room_label"):
                if str_field in entry and isinstance(entry[str_field], str):
                    obs[str_field] = entry[str_field]
            if "objects_detected" in entry:
                obj_list = entry["objects_detected"]
                if isinstance(obj_list, list):
                    obs["objects_detected"] = [
                        str(o) for o in obj_list if isinstance(o, str)
                    ]
            obs["ts"] = self._utc_now_iso()
            validated.append(obs)

        if not os.path.isfile(spatial_memory_file):
            memory = {
                "task_id": "",
                "grid_resolution_m": 0.5,
                "snapshots": [],
                "rooms": {},
                "object_sightings": {},
            }
        else:
            with open(spatial_memory_file, "r", encoding="utf-8") as f:
                memory = json.load(f)

        snapshots = memory.get("snapshots", [])
        rooms = memory.get("rooms", {})
        object_sightings = memory.get("object_sightings", {})

        for entry in validated:
            snapshots.append(entry)
            room = entry.get("room_label")
            if room and isinstance(room, str):
                rooms.setdefault(room, {"first_seen": entry["ts"], "visit_count": 0})
                rooms[room]["visit_count"] = rooms[room].get("visit_count", 0) + 1
                rooms[room]["last_seen"] = entry["ts"]
            for obj_name in entry.get("objects_detected", []):
                object_sightings.setdefault(obj_name, {"count": 0})
                object_sightings[obj_name]["count"] = (
                    object_sightings[obj_name].get("count", 0) + 1
                )
                object_sightings[obj_name]["last_seen"] = entry["ts"]

        if len(snapshots) > self._SPATIAL_MEMORY_MAX_SNAPSHOTS:
            snapshots = snapshots[-self._SPATIAL_MEMORY_MAX_SNAPSHOTS :]

        memory["snapshots"] = snapshots
        memory["rooms"] = rooms
        memory["object_sightings"] = object_sightings
        self._persist_json_atomic(spatial_memory_file, memory)

    @staticmethod
    def _resolve_nav_loop_script() -> str:
        candidates: list[str] = []
        env_override = os.environ.get("OPENCLAW_NAV_LOOP_SCRIPT")
        if isinstance(env_override, str) and env_override:
            candidates.append(env_override)
        candidates.append(_NAV_LOOP_SCRIPT)

        module_dir = os.path.dirname(os.path.abspath(__file__))
        cursor = module_dir
        for _ in range(8):
            candidates.append(os.path.join(cursor, "tools", "nav_agent.py"))
            parent = os.path.dirname(cursor)
            if parent == cursor:
                break
            cursor = parent

        candidates.append(
            os.path.abspath(os.path.join(os.getcwd(), "tools", "nav_agent.py"))
        )

        checked: list[str] = []
        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if normalized in checked:
                continue
            checked.append(normalized)
            if os.path.exists(normalized):
                return normalized

        raise HabitatAdapterError(
            "nav_agent.py not found; checked: " + ", ".join(checked)
        )

    def _read_nav_status_file(
        self, nav_status_file: str
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            with open(nav_status_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return None, str(exc)
        if not isinstance(data, dict):
            return None, "nav_status.json root must be a JSON object"
        return data, None

    @staticmethod
    def _is_terminal_nav_status(status: Any) -> bool:
        return (
            isinstance(status, str)
            and status in _NAV_STATUS_ALLOWED_VALUES
            and status != "in_progress"
        )

    def _force_nav_loop_terminal_status(
        self,
        nav_status: Mapping[str, Any],
        *,
        returncode: int,
        reason: str,
    ) -> tuple[Dict[str, Any], bool]:
        updated = copy.deepcopy(dict(nav_status))
        if self._is_terminal_nav_status(updated.get("status")):
            return updated, False

        updated["status"] = "error"
        existing_error = updated.get("error")
        if not (isinstance(existing_error, str) and existing_error.strip()):
            updated["error"] = (
                "nav-loop process exited before reaching a terminal state: "
                f"returncode={returncode}, reason={reason}"
            )
        updated["updated_at"] = self._utc_now_iso()
        state_version = updated.get("state_version")
        if (
            isinstance(state_version, int)
            and not isinstance(state_version, bool)
            and state_version >= 0
        ):
            updated["state_version"] = state_version + 1
        else:
            updated["state_version"] = 1
        return updated, True

    @staticmethod
    def _action_history_signature(entry: Any) -> tuple[Any, ...]:
        if not isinstance(entry, Mapping):
            return ("raw", str(entry))

        pos_value = entry.get("pos")
        pos_sig: Optional[tuple[float, float, float]]
        if isinstance(pos_value, (list, tuple)) and len(pos_value) == 3:
            try:
                pos_sig = (
                    round(float(pos_value[0]), 6),
                    round(float(pos_value[1]), 6),
                    round(float(pos_value[2]), 6),
                )
            except (TypeError, ValueError):
                pos_sig = None
        else:
            pos_sig = None

        return (
            entry.get("step"),
            entry.get("action"),
            pos_sig,
            entry.get("collided"),
            entry.get("nav_status"),
            entry.get("saw"),
        )

    def _build_action_history_entry_from_state(
        self,
        nav_status: Mapping[str, Any],
        sim_step_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a fallback action_history entry from nav_status fields.

        Only produces an entry when last_action contains a meaningful action
        name — otherwise returns None to avoid ghost entries with just a step
        number and position.
        """
        entry: Dict[str, Any] = {}

        last_action = nav_status.get("last_action")
        if not isinstance(last_action, Mapping):
            return None

        action_value = last_action.get("action")
        if action_value is None:
            action_value = last_action.get("last_action")
        if not action_value:
            # No meaningful action to record — skip fallback entirely
            return None
        entry["action"] = str(action_value)

        # Use step from last_action if available, then sim_step_count
        # (the authoritative counter matching image filenames).
        step_value = last_action.get("step")
        if step_value is None and sim_step_count is not None:
            step_value = sim_step_count
        if step_value is not None:
            entry["step"] = self._coerce_non_negative_int(
                step_value, field_name="payload.patch.last_action.step"
            )

        if "collided" in last_action:
            entry["collided"] = self._coerce_bool(
                last_action.get("collided"),
                field_name="payload.patch.last_action.collided",
            )

        nav_state_value = last_action.get("nav_status")
        if nav_state_value is not None:
            entry["nav_status"] = str(nav_state_value)

        if "reachable" in last_action:
            entry["reachable"] = self._coerce_bool(
                last_action.get("reachable"),
                field_name="payload.patch.last_action.reachable",
            )

        saw_value = last_action.get("saw")
        if saw_value is not None:
            entry["saw"] = str(saw_value)

        current_position = nav_status.get("current_position")
        if current_position is not None:
            entry["pos"] = self._coerce_float_list(
                current_position, 3, field_name="payload.patch.current_position"
            )

        geodesic = nav_status.get("geodesic_distance")
        if geodesic is not None:
            entry["geodesic_distance"] = self._coerce_optional_float(
                geodesic, field_name="payload.patch.geodesic_distance"
            )

        return entry

    def _normalize_action_history_entry(
        self,
        entry: Mapping[str, Any],
        nav_status: Mapping[str, Any],
        index: int,
        sim_step_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = copy.deepcopy(dict(entry))
        field_prefix = f"payload.patch.action_history_append[{index}]"

        step_value = normalized.get("step")
        if step_value is None and "step_count" in normalized:
            step_value = normalized.get("step_count")
        # Fall back to simulator step_count — the authoritative counter
        # that matches image filenames ({session_id}_step{NNNNNN}_*.png).
        if step_value is None and sim_step_count is not None:
            step_value = sim_step_count
        if step_value is not None:
            normalized["step"] = self._coerce_non_negative_int(
                step_value, field_name=f"{field_prefix}.step"
            )
        normalized.pop("step_count", None)

        action_value = normalized.get("action")
        if action_value is None:
            action_value = normalized.get("last_action")
        if action_value is None:
            # Fallback: try nav_status.last_action (may be string or dict)
            la = nav_status.get("last_action")
            if isinstance(la, str) and la.strip():
                action_value = la.strip()
            elif isinstance(la, Mapping):
                action_value = la.get("action") or la.get("last_action")
        if action_value is not None:
            normalized["action"] = str(action_value)
        normalized.pop("last_action", None)

        pos_value = normalized.get("pos")
        if pos_value is None:
            pos_value = normalized.get("position")
        if pos_value is None:
            pos_value = nav_status.get("current_position")
        # Fallback to GT position from _debug (always injected, even mapless).
        # This ensures TUI developers see pos in action_history regardless of
        # nav mode — the agent never reads raw action_history entries.
        if pos_value is None:
            debug = nav_status.get("_debug", {})
            if isinstance(debug, dict):
                pos_value = debug.get("gt_position")
        if pos_value is not None:
            normalized["pos"] = self._coerce_float_list(
                pos_value, 3, field_name=f"{field_prefix}.pos"
            )
        normalized.pop("position", None)

        collided_value = normalized.get("collided")
        if collided_value is None and isinstance(nav_status.get("last_action"), Mapping):
            la = nav_status["last_action"]
            if "collided" in la:
                collided_value = la.get("collided")
        if collided_value is not None:
            normalized["collided"] = self._coerce_bool(
                collided_value, field_name=f"{field_prefix}.collided"
            )

        if "reachable" in normalized and normalized.get("reachable") is not None:
            normalized["reachable"] = self._coerce_bool(
                normalized["reachable"], field_name=f"{field_prefix}.reachable"
            )

        if "geodesic_distance" in normalized:
            normalized["geodesic_distance"] = self._coerce_optional_float(
                normalized["geodesic_distance"],
                field_name=f"{field_prefix}.geodesic_distance",
            )
        elif nav_status.get("geodesic_distance") is not None:
            normalized["geodesic_distance"] = self._coerce_optional_float(
                nav_status.get("geodesic_distance"),
                field_name=f"{field_prefix}.geodesic_distance",
            )

        # Auto-populate heading from _debug if available
        if "heading_deg" not in normalized:
            debug = nav_status.get("_debug", {})
            if isinstance(debug, dict) and debug.get("gt_heading_deg") is not None:
                normalized["heading_deg"] = debug["gt_heading_deg"]

        # Structured reasoning fields (perception → analysis → decision)
        for reasoning_key in ("perception", "analysis", "decision"):
            if reasoning_key in normalized and normalized[reasoning_key] is not None:
                normalized[reasoning_key] = str(normalized[reasoning_key])

        # Backward compatibility: populate saw from perception if not provided
        if "saw" not in normalized:
            if "perception" in normalized and normalized["perception"]:
                normalized["saw"] = normalized["perception"]
            else:
                for key in ("scene_description", "summary"):
                    value = normalized.get(key)
                    if isinstance(value, str) and value:
                        normalized["saw"] = value
                        break
        if "saw" in normalized and normalized["saw"] is not None:
            normalized["saw"] = str(normalized["saw"])

        # Attach the current visual path so developers can trace which image
        # the agent was looking at for this action entry.
        if "visual_path" not in normalized:
            visual = nav_status.get("last_visual")
            vpath = None
            if isinstance(visual, Mapping):
                vpath = visual.get("path")
            elif isinstance(visual, str) and visual.startswith("/"):
                vpath = visual
            if isinstance(vpath, str) and vpath.startswith("/"):
                normalized["visual_path"] = vpath

        if "nav_status" in normalized and normalized["nav_status"] is not None:
            normalized["nav_status"] = str(normalized["nav_status"])

        if not any(
            key in normalized
            for key in (
                "step",
                "action",
                "pos",
                "collided",
                "nav_status",
                "geodesic_distance",
                "reachable",
                "saw",
                "perception",
                "analysis",
                "decision",
            )
        ):
            fallback = self._build_action_history_entry_from_state(nav_status)
            if fallback is not None:
                normalized.update(fallback)

        return normalized

    def _append_action_history_entries(
        self,
        history: list[Any],
        entries: list[Mapping[str, Any]],
    ) -> list[Any]:
        merged = copy.deepcopy(history)
        for item in entries:
            if (
                merged
                and self._action_history_signature(merged[-1])
                == self._action_history_signature(item)
            ):
                continue
            merged.append(copy.deepcopy(dict(item)))
        if len(merged) > _MAX_NAV_ACTION_HISTORY:
            merged = merged[-_MAX_NAV_ACTION_HISTORY:]
        return merged

    @staticmethod
    def _extract_visual_path(value: Any) -> Optional[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        if isinstance(value, Mapping):
            path_value = value.get("path")
            if isinstance(path_value, str):
                normalized = path_value.strip()
                return normalized or None
        return None

    @staticmethod
    def _extract_saw_text(value: Any) -> Optional[str]:
        if not isinstance(value, Mapping):
            return None
        for key in ("perception", "saw", "scene_description", "summary"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _is_motion_action(action: Any) -> bool:
        if not isinstance(action, str):
            return False
        normalized = action.strip().lower()
        if not normalized:
            return False
        if re.search(
            r"(^|[^a-z])(look|observe|observation|metrics|status|inspect|image|analy[sz]e|analysis)([^a-z]|$)",
            normalized,
        ):
            return False
        return bool(
            re.search(
                r"(^|[^a-z])(move|forward|backward|strafe|turn|rotate|navigate|step)([^a-z]|$)",
                normalized,
            )
        )

    def _patch_contains_mapless_motion(
        self,
        current_nav_status: Mapping[str, Any],
        patch: Mapping[str, Any],
        updated_nav_status: Mapping[str, Any],
    ) -> bool:
        previous_steps = current_nav_status.get("total_steps")
        next_steps = updated_nav_status.get("total_steps")
        if (
            isinstance(previous_steps, int)
            and not isinstance(previous_steps, bool)
            and previous_steps >= 0
            and isinstance(next_steps, int)
            and not isinstance(next_steps, bool)
            and next_steps >= 0
            and next_steps > previous_steps
        ):
            return True

        action_append = patch.get("action_history_append")
        if isinstance(action_append, list):
            for entry in action_append:
                if not isinstance(entry, Mapping):
                    continue
                if self._is_motion_action(entry.get("action")) or self._is_motion_action(
                    entry.get("last_action")
                ):
                    return True

        return False

    def _validate_mapless_visual_grounding_patch(
        self,
        *,
        current_nav_status: Mapping[str, Any],
        patch: Mapping[str, Any],
        updated_nav_status: Mapping[str, Any],
    ) -> None:
        if updated_nav_status.get("nav_mode") != "mapless":
            return
        if not self._patch_contains_mapless_motion(
            current_nav_status=current_nav_status,
            patch=patch,
            updated_nav_status=updated_nav_status,
        ):
            return

        if "last_visual" not in patch:
            raise HabitatAdapterError(
                'Mapless motion patch must include "last_visual" from a look observation before movement'
            )
        if self._extract_visual_path(patch.get("last_visual")) is None:
            raise HabitatAdapterError(
                'Field "payload.patch.last_visual" must include a non-empty image path for mapless motion updates'
            )

        action_append = patch.get("action_history_append")
        if not isinstance(action_append, list) or not action_append:
            raise HabitatAdapterError(
                'Mapless motion patch must include non-empty "action_history_append" with visual reasoning (perception/saw)'
            )

        has_saw = any(
            self._extract_saw_text(entry) is not None for entry in action_append
        )
        if not has_saw:
            raise HabitatAdapterError(
                'Mapless motion patch must include at least one "action_history_append" entry with non-empty "perception" or "saw"'
            )

    def _apply_nav_status_patch(
        self,
        nav_status: Mapping[str, Any],
        patch: Mapping[str, Any],
        sim_step_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        immutable_keys = sorted(set(patch).intersection(_NAV_STATUS_IMMUTABLE_FIELDS))
        if immutable_keys:
            keys = ", ".join(immutable_keys)
            raise HabitatAdapterError(f"Patch contains immutable fields: {keys}")

        unknown_keys = sorted(set(patch).difference(_NAV_STATUS_PATCH_FIELDS))
        if unknown_keys:
            keys = ", ".join(unknown_keys)
            raise HabitatAdapterError(f"Patch contains unsupported fields: {keys}")

        updated = copy.deepcopy(dict(nav_status))

        if "substeps" in patch:
            substeps = patch["substeps"]
            if not isinstance(substeps, list):
                raise HabitatAdapterError('Field "payload.patch.substeps" must be a list')
            updated["substeps"] = copy.deepcopy(substeps)

        if "current_substep_index" in patch:
            updated["current_substep_index"] = self._coerce_non_negative_int(
                patch["current_substep_index"],
                "payload.patch.current_substep_index",
            )

        if "status" in patch:
            status = patch["status"]
            if not isinstance(status, str) or status not in _NAV_STATUS_ALLOWED_VALUES:
                raise HabitatAdapterError(
                    'Field "payload.patch.status" must be one of '
                    "in_progress|reached|blocked|error|timeout"
                )
            updated["status"] = status

        if "nav_phase" in patch:
            nav_phase = patch["nav_phase"]
            if not isinstance(nav_phase, str) or not nav_phase:
                raise HabitatAdapterError(
                    'Field "payload.patch.nav_phase" must be a non-empty str'
                )
            updated["nav_phase"] = nav_phase

        if "total_steps" in patch:
            updated["total_steps"] = self._coerce_non_negative_int(
                patch["total_steps"],
                "payload.patch.total_steps",
            )

        if "collisions" in patch:
            updated["collisions"] = self._coerce_non_negative_int(
                patch["collisions"],
                "payload.patch.collisions",
            )

        if "current_position" in patch:
            current_position = patch["current_position"]
            if current_position is None:
                updated["current_position"] = None
            else:
                updated["current_position"] = self._coerce_float_list(
                    current_position, 3, "payload.patch.current_position"
                )

        if "geodesic_distance" in patch:
            updated["geodesic_distance"] = self._coerce_optional_float(
                patch["geodesic_distance"],
                "payload.patch.geodesic_distance",
            )

        if "rooms_discovered" in patch:
            rooms_discovered = patch["rooms_discovered"]
            if not isinstance(rooms_discovered, list):
                raise HabitatAdapterError(
                    'Field "payload.patch.rooms_discovered" must be a list'
                )
            updated["rooms_discovered"] = copy.deepcopy(rooms_discovered)

        if "last_visual" in patch:
            updated["last_visual"] = copy.deepcopy(patch["last_visual"])

        if "last_action" in patch:
            updated["last_action"] = copy.deepcopy(patch["last_action"])

        history = updated.get("action_history")
        if not isinstance(history, list):
            history = []
        normalized_append: list[Mapping[str, Any]] = []

        if "action_history_append" in patch:
            action_append = patch["action_history_append"]
            if not isinstance(action_append, list):
                raise HabitatAdapterError(
                    'Field "payload.patch.action_history_append" must be a list'
                )
            for index, item in enumerate(action_append):
                if not isinstance(item, Mapping):
                    raise HabitatAdapterError(
                        f'Field "payload.patch.action_history_append[{index}]" must be an object'
                    )
                normalized_append.append(
                    self._normalize_action_history_entry(
                        item, updated, index, sim_step_count=sim_step_count
                    )
                )
        elif any(
            field in patch
            for field in (
                "last_action",
                "current_position",
                "total_steps",
                "geodesic_distance",
                "status",
            )
        ):
            fallback_entry = self._build_action_history_entry_from_state(
                updated, sim_step_count=sim_step_count
            )
            if fallback_entry is not None:
                normalized_append.append(fallback_entry)

        if normalized_append:
            history = self._append_action_history_entries(history, normalized_append)
            updated["action_history"] = history

        if "spatial_memory_file" in patch:
            spatial_memory_file = patch["spatial_memory_file"]
            if not isinstance(spatial_memory_file, str) or not spatial_memory_file:
                raise HabitatAdapterError(
                    'Field "payload.patch.spatial_memory_file" must be a non-empty str'
                )
            updated["spatial_memory_file"] = spatial_memory_file

        if "spatial_memory_append" in patch:
            self._apply_spatial_memory_append(
                patch["spatial_memory_append"],
                updated.get("spatial_memory_file", ""),
            )

        if "finding" in patch:
            updated["finding"] = copy.deepcopy(patch["finding"])

        if "error" in patch:
            error_value = patch["error"]
            if error_value is not None and not isinstance(error_value, str):
                raise HabitatAdapterError(
                    'Field "payload.patch.error" must be str or null'
                )
            updated["error"] = error_value

        if "capability_request" in patch:
            cap_req = patch["capability_request"]
            if cap_req is not None and not isinstance(cap_req, str):
                raise HabitatAdapterError(
                    'Field "payload.patch.capability_request" must be str or null'
                )
            updated["capability_request"] = cap_req

        return updated

    @staticmethod
    def _maybe_update_int(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = HabitatAdapterPatchMixin._coerce_int(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _maybe_update_float(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = HabitatAdapterPatchMixin._coerce_float(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _maybe_update_bool(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = HabitatAdapterPatchMixin._coerce_bool(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _coerce_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise HabitatAdapterError(f'Field "{field_name}" must be int, got bool')
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HabitatAdapterError(f'Field "{field_name}" must be int') from exc

    @staticmethod
    def _coerce_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise HabitatAdapterError(
                f'Field "{field_name}" must be float, got bool'
            )
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise HabitatAdapterError(f'Field "{field_name}" must be float') from exc

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, np.integer)):
            return bool(value)
        raise HabitatAdapterError(f'Field "{field_name}" must be bool')

    @staticmethod
    def _coerce_float_list(value: Any, expected_len: int, field_name: str) -> list[float]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != expected_len:
            raise HabitatAdapterError(
                f'Field "{field_name}" must be list[float] length {expected_len}'
            )
        return [
            HabitatAdapterPatchMixin._coerce_float(item, field_name=field_name)
            for item in value
        ]
