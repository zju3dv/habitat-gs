"""Long-running nav-loop orchestration for the habitat-gs adapter.

State ownership (read carefully before adding writers)
======================================================

nav_status.json (on disk, per loop)
    OWNER: bridge process.
    WRITERS:
        - _persist_json_atomic via _start_nav_loop (initial write),
          _update_nav_loop_status (patch merge), _finalize_nav_loop_record
          (terminal force write), and _get_nav_loop_status (refresh path,
          ACTIVE loops only).
        - All bridge writers acquire _nav_loop_lock and use atomic
          temp-file + rename + fsync.
    FALLBACK WRITER:
        - nav_agent subprocess via tools/nav_agent.py:mark_terminal_status
          ONLY when its bridge RPC fails (path 1 = bridge, path 2 = locked
          local fallback). The fallback acquires a sidecar flock so it
          serializes with bridge writers across processes.

record.nav_status (in-memory, per loop on the bridge)
    OWNER: bridge.
    Updated under _nav_loop_lock by _update_nav_loop_status and
    _finalize_nav_loop_record. _get_nav_loop_status with
    include_nav_status=True refreshes _debug from session state ONLY for
    ACTIVE loops; closed loops keep their terminal snapshot to avoid
    rewriting historical metrics with current session state.

session.{cumulative_path_length, trajectory, last_goal, eval_goal,
         initial_geodesic_distance, mapless}
    OWNER: bridge.
    _start_nav_loop owns the per-loop reset of these fields. The reset is
    transactional: a "pending" phase computes new values into locals,
    a "tentative commit" phase applies them while a rollback snapshot
    captures the originals, and any failure (validation, persistence,
    Popen, registration) triggers rollback so the session is byte-identical
    to its pre-call state. _record_pose owns per-step path accumulation.
    No other code path mutates these fields outside the rollback envelope.

The single-owner principle (bridge writes, subprocess goes through bridge
RPC) prevents the cross-process read-modify-write races identified in the
PR #28 audit (B1, B5). The flock fallback covers the desperate case where
the bridge RPC is unreachable and the subprocess MUST persist a terminal
status before exiting.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Mapping, Optional

from .types import (
    _MAX_CLOSED_NAV_LOOPS,
    _NavLoopRecord,
    _Session,
    HabitatAdapterError,
)

# Accepts plain IPv4 addresses and valid hostnames; no IPv6 support.
_GATEWAY_HOST_RE = re.compile(
    r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
    r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*'
    r'|(?:[0-9]{1,3}\.){3}[0-9]{1,3})$'
)


class HabitatAdapterNavLoopMixin:
    """Long-running nav-loop orchestration and policy enforcement."""

    @staticmethod
    def _coerce_success_threshold(value: Any) -> Optional[float]:
        """Coerce a user-provided success_distance_threshold to a positive
        float, or return None for any invalid input.

        Excludes bool explicitly: bool is a subclass of int in Python, so
        a plain isinstance(value, (int, float)) check would silently accept
        True/False and serialize them as 1.0/0.0, widening the success
        threshold for mis-typed JSON callers and corrupting benchmark results.
        """
        if isinstance(value, bool):
            return None
        if not isinstance(value, (int, float)):
            return None
        if value <= 0:
            return None
        return float(value)

    def _compute_geodesic_distance(
        self,
        session: _Session,
        start_pos: Any,
        end_pos: Any,
    ) -> Optional[float]:
        """Compute geodesic distance between two 3D points, snapping to navmesh.

        Returns None if pathfinder unavailable, points unreachable, or any
        numerical error occurs.
        """
        pathfinder = self._get_pathfinder(session)
        if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
            return None
        try:
            import habitat_sim as _hsim
            import numpy as _np
            start = _np.asarray(start_pos, dtype=_np.float32)
            end = _np.asarray(end_pos, dtype=_np.float32)
            # Snap both points to navmesh — without this, goals not exactly
            # on navmesh vertices cause find_path to fail silently.
            start = _np.asarray(pathfinder.snap_point(start), dtype=_np.float32)
            end = _np.asarray(pathfinder.snap_point(end), dtype=_np.float32)
            sp = _hsim.ShortestPath()
            sp.requested_start = start
            sp.requested_end = end
            if pathfinder.find_path(sp):
                return float(sp.geodesic_distance)
        except Exception:
            pass
        return None

    def _build_debug_snapshot(self, session: _Session) -> Dict[str, Any]:
        """Build a complete _debug snapshot from current session state.

        Reads GT goal from session.eval_goal (independent of agent-facing
        session.last_goal) so non-pointnav tasks with evaluation GT do NOT
        leak goal coordinates or polar signals to the agent.

        Called from _start_nav_loop, update_nav_loop_status, and refresh
        paths (get_nav_loop_status) so the snapshot is always fresh.
        """
        import numpy as _np

        position_np = self._current_position(session)
        position = position_np.tolist() if hasattr(position_np, "tolist") else list(position_np)
        heading_deg = round(self._heading_degrees(session), 3)

        eval_goal = session.eval_goal
        gt_goal = list(eval_goal) if eval_goal is not None else None

        gt_euclidean = None
        gt_geodesic = None
        gt_direction = None
        if eval_goal is not None:
            try:
                goal_arr = _np.asarray(eval_goal, dtype=_np.float32)
                pos_arr = _np.asarray(position, dtype=_np.float32)
                delta = goal_arr - pos_arr
                gt_euclidean = round(float(_np.linalg.norm(delta)), 3)
                goal_bearing = float(_np.degrees(_np.arctan2(delta[2], delta[0])))
                rel = goal_bearing - heading_deg
                rel = (rel + 180.0) % 360.0 - 180.0
                gt_direction = round(rel, 1)
            except Exception:
                pass
            d = self._compute_geodesic_distance(session, position, list(eval_goal))
            if d is not None:
                gt_geodesic = round(d, 3)

        return {
            "gt_position": position,
            "gt_heading_deg": heading_deg,
            "gt_goal": gt_goal,
            "gt_euclidean_distance": gt_euclidean,
            "gt_geodesic_distance": gt_geodesic,
            "gt_goal_direction_deg": gt_direction,
            "gt_initial_geodesic_distance": (
                round(session.initial_geodesic_distance, 3)
                if session.initial_geodesic_distance is not None
                else None
            ),
            "gt_path_length": round(session.cumulative_path_length, 3),
        }

    def _start_nav_loop(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id

        _SUPPORTED_TASK_TYPES = ("pointnav", "objectnav", "imagenav", "instruction_following", "eqa")
        task_type = payload.get("task_type", "pointnav")
        if task_type not in _SUPPORTED_TASK_TYPES:
            raise HabitatAdapterError(
                f'Field "payload.task_type" must be one of {_SUPPORTED_TASK_TYPES}'
            )

        nav_mode_raw = payload.get("nav_mode")
        if nav_mode_raw is not None and nav_mode_raw not in ("navmesh", "mapless"):
            raise HabitatAdapterError(
                'Field "payload.nav_mode" must be "navmesh" or "mapless"'
            )

        goal_type = payload.get("goal_type")
        if goal_type not in ("position", "instruction"):
            raise HabitatAdapterError(
                'Field "payload.goal_type" must be "position" or "instruction"'
            )

        goal_description = payload.get("goal_description")
        if not isinstance(goal_description, str) or not goal_description:
            raise HabitatAdapterError(
                'Field "payload.goal_description" must be a non-empty str'
            )

        goal_position_raw = payload.get("goal_position")
        goal_position = None
        if goal_position_raw is not None:
            goal_position = self._coerce_float_list(
                goal_position_raw, 3, "payload.goal_position"
            )
        if goal_type == "position" and goal_position is None:
            raise HabitatAdapterError(
                'Field "payload.goal_position" is required when goal_type is "position"'
            )

        # Evaluation ground-truth goal: separate from goal_position so that
        # non-pointnav tasks can provide GT without leaking it to the agent.
        eval_goal_raw = payload.get("eval_goal_position")
        eval_goal = None
        if eval_goal_raw is not None:
            eval_goal = self._coerce_float_list(
                eval_goal_raw, 3, "payload.eval_goal_position"
            )

        # If caller explicitly claims has_ground_truth=True, an eval_goal_position
        # is required. Without this check the flag would be silently dropped
        # (nav_status.has_ground_truth is derived from session.eval_goal, not
        # from payload), causing benchmark harnesses to think they marked GT
        # when they actually did not. For pointnav, goal_position acts as the
        # fallback eval_goal so the flag is compatible with a bare goal_position.
        has_ground_truth_flag = self._coerce_bool(
            payload.get("has_ground_truth", False),
            field_name="payload.has_ground_truth",
        )
        if (
            has_ground_truth_flag
            and eval_goal is None
            and not (task_type == "pointnav" and goal_position is not None)
        ):
            raise HabitatAdapterError(
                'Field "payload.has_ground_truth" is True but no '
                '"eval_goal_position" was provided. Either pass '
                '"eval_goal_position" explicitly or (for pointnav) '
                '"goal_position", or set has_ground_truth=False.'
            )

        reference_image = payload.get("reference_image")
        if reference_image is not None and not isinstance(reference_image, str):
            raise HabitatAdapterError(
                'Field "payload.reference_image" must be a string path'
            )

        bridge_session_id = payload.get("session_id")
        if not isinstance(bridge_session_id, str) or not bridge_session_id:
            raise HabitatAdapterError(
                'Field "payload.session_id" must be a non-empty str'
            )

        # Reject starting a second loop on the same session
        with self._nav_loop_lock:
            for lid, rec in self._nav_loops.items():
                if rec.session_id == bridge_session_id and rec.process.poll() is None:
                    raise HabitatAdapterError(
                        f'Session "{bridge_session_id}" already has an active nav loop '
                        f'"{lid}". Stop it first or use a different session.'
                    )

        session = self._require_session(bridge_session_id)
        pathfinder = self._get_pathfinder(session)
        has_navmesh = bool(
            getattr(pathfinder, "is_loaded", False) if pathfinder is not None else False
        )
        # Keep scene capability and agent policy separate: mapless forbids
        # map-planning tools for the agent, but the system may still have navmesh.
        nav_mode = nav_mode_raw or ("navmesh" if has_navmesh else "mapless")

        if payload.get("gateway_host") is not None:
            gateway_host = str(payload.get("gateway_host"))
        else:
            gateway_host = os.environ.get("OPENCLAW_GATEWAY_HOST", "").strip() or "127.0.0.1"
        if not _GATEWAY_HOST_RE.match(gateway_host):
            raise HabitatAdapterError(
                'Field "payload.gateway_host" must be a valid hostname or IPv4 address'
            )
        if payload.get("gateway_port") is not None:
            gateway_port = self._coerce_int(payload.get("gateway_port"), "payload.gateway_port")
        else:
            env_port = os.environ.get("OPENCLAW_GATEWAY_PORT", "").strip()
            gateway_port = (
                self._coerce_int(env_port, "OPENCLAW_GATEWAY_PORT")
                if env_port
                else 18789
            )
        if not (1 <= gateway_port <= 65535):
            raise HabitatAdapterError('Field "payload.gateway_port" must be in range 1..65535')
        max_iterations = self._coerce_int(payload.get("max_iterations", 50), "payload.max_iterations")
        if not (1 <= max_iterations <= 10000):
            raise HabitatAdapterError('Field "payload.max_iterations" must be in range 1..10000')
        agent_timeout = self._coerce_int(payload.get("agent_timeout", 120), "payload.agent_timeout")
        if not (1 <= agent_timeout <= 86400):
            raise HabitatAdapterError('Field "payload.agent_timeout" must be in range 1..86400')

        workspace_host_payload = payload.get("workspace_host")
        if workspace_host_payload is None:
            workspace_host = str(os.environ.get("NAV_WORKSPACE_HOST", "")).strip()
        else:
            if not isinstance(workspace_host_payload, str) or not workspace_host_payload.strip():
                raise HabitatAdapterError(
                    'Field "payload.workspace_host" must be a non-empty str'
                )
            workspace_host = workspace_host_payload.strip()

        loop_id = f"navloop-{uuid.uuid4().hex[:8]}"
        from .types import _DEFAULT_VISUAL_OUTPUT_DIR
        artifacts_base = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        loop_dir = os.path.join(artifacts_base, bridge_session_id, loop_id)
        os.makedirs(loop_dir, exist_ok=True)
        nav_status_file = os.path.join(loop_dir, "nav_status.json")
        abs_nav_status_file = os.path.abspath(nav_status_file)
        for marker in ("/data/nav_artifacts/", "/artifacts/habitat-gs/", "/artifacts/"):
            if not workspace_host:
                marker_index = abs_nav_status_file.find(marker)
                if marker_index > 0:
                    workspace_host = abs_nav_status_file[:marker_index]
        if not workspace_host:
            workspace_host = os.getcwd()

        nav_status_dir = os.path.dirname(abs_nav_status_file)
        os.makedirs(nav_status_dir, exist_ok=True)
        spatial_memory_file = os.path.join(nav_status_dir, "spatial_memory.json")

        # Compute the derived state into LOCAL variables first. Session
        # fields are mutated only after every validation (including the
        # reachability check below) has passed — otherwise a raise here
        # leaves the live session with a stale goal + zeroed path_length
        # from a loop that never actually started.
        current_pos_list = self._current_position(session).tolist()

        # Agent-facing goal (pointnav only — drives state_summary polar
        # signals visible to the agent). For non-pointnav (or any task
        # without an explicit goal_position), the agent must NOT see polar
        # signals derived from a stale prior-loop goal, so the pending
        # value is explicitly None and the commit phase will overwrite
        # session.last_goal unconditionally.
        pending_last_goal: Optional[list[float]]
        if goal_position is not None:
            pending_last_goal = [float(v) for v in goal_position]
        else:
            pending_last_goal = None

        # Evaluation ground-truth goal (invisible to agent — drives _debug
        # snapshot and SPL computation only). For pointnav, falls back to
        # goal_position if eval_goal is not explicitly provided.
        pending_eval_goal: Optional[list[float]]
        if eval_goal is not None:
            pending_eval_goal = [float(v) for v in eval_goal]
        elif goal_position is not None and task_type == "pointnav":
            pending_eval_goal = [float(v) for v in goal_position]
        else:
            pending_eval_goal = None

        # Per-loop policy state. mapless flag belongs to the session
        # because navigation tools (forward / turn / look) consult it to
        # decide whether to expose absolute coordinates. It used to be
        # set in a post-Popen block that the rollback could not protect.
        pending_mapless: bool = (nav_mode == "mapless")

        # Compute initial geodesic distance (l_opt) from eval_goal. Note:
        # _compute_geodesic_distance only reads the pathfinder from session,
        # not session.eval_goal, so passing pending_eval_goal is correct.
        pending_initial_geodesic: Optional[float] = None
        if pending_eval_goal is not None:
            pending_initial_geodesic = self._compute_geodesic_distance(
                session, current_pos_list, pending_eval_goal
            )
            # Reject unreachable goals upfront: if a navmesh is loaded and
            # the goal has no path from the current position, there is no
            # point starting a nav loop — SPL/success can never be valid.
            # Only enforced when navmesh is available; mapless scenes legitimately
            # have no pathfinder so we cannot pre-check reachability there.
            if has_navmesh and pending_initial_geodesic is None:
                raise HabitatAdapterError(
                    f"eval_goal_position {pending_eval_goal} is unreachable "
                    f"from the agent's current position on the navmesh. "
                    "Pick a navigable goal or disable GT evaluation."
                )

        # Snapshot the pre-mutation state so we can roll back on any
        # failure between here and subprocess.Popen() + record registration
        # succeeding. Without rollback, a disk-full during _persist_json_atomic
        # or a Popen OSError would leave session state referencing a loop
        # that never actually ran, corrupting subsequent get_metrics / SPL
        # readings.
        _rollback_snapshot = {
            "cumulative_path_length": session.cumulative_path_length,
            "trajectory": session.trajectory,
            "last_goal": session.last_goal,
            "eval_goal": session.eval_goal,
            "initial_geodesic_distance": session.initial_geodesic_distance,
            "mapless": session.mapless,
        }

        # Tentatively commit the mutations — _build_debug_snapshot below
        # reads from session state (eval_goal, cumulative_path_length,
        # initial_geodesic_distance) so the new values must be visible
        # when the initial debug snapshot is constructed. If anything
        # downstream fails, the except block restores the snapshot.
        session.cumulative_path_length = 0.0
        session.trajectory = [current_pos_list]
        # last_goal is committed UNCONDITIONALLY (including the None case)
        # so the agent sees a fresh polar context for this loop, never a
        # stale one from a previous loop on the same session.
        session.last_goal = pending_last_goal
        session.eval_goal = pending_eval_goal
        session.initial_geodesic_distance = pending_initial_geodesic
        session.mapless = pending_mapless

        try:
            return self._launch_nav_loop_process(
                session=session,
                bridge_session_id=bridge_session_id,
                loop_id=loop_id,
                task_type=task_type,
                nav_mode=nav_mode,
                has_navmesh=has_navmesh,
                goal_type=goal_type,
                goal_description=goal_description,
                goal_position=goal_position,
                reference_image=reference_image,
                payload=payload,
                nav_status_file=nav_status_file,
                spatial_memory_file=spatial_memory_file,
                workspace_host=workspace_host,
                max_iterations=max_iterations,
                agent_timeout=agent_timeout,
            )
        except BaseException:
            # Roll back the tentative session mutations — the loop never
            # started, so from the caller's perspective the session must
            # look exactly as it did before this call. mapless was added
            # to the snapshot in Phase 5 (S3 fix); the kill-orphan logic
            # for post-Popen failures lives inside _launch_nav_loop_process.
            session.cumulative_path_length = _rollback_snapshot["cumulative_path_length"]
            session.trajectory = _rollback_snapshot["trajectory"]
            session.last_goal = _rollback_snapshot["last_goal"]
            session.eval_goal = _rollback_snapshot["eval_goal"]
            session.initial_geodesic_distance = _rollback_snapshot["initial_geodesic_distance"]
            session.mapless = _rollback_snapshot["mapless"]
            raise

    def _launch_nav_loop_process(
        self,
        *,
        session: _Session,
        bridge_session_id: str,
        loop_id: str,
        task_type: str,
        nav_mode: str,
        has_navmesh: bool,
        goal_type: str,
        goal_description: str,
        goal_position: Optional[list[float]],
        reference_image: Any,
        payload: Mapping[str, Any],
        nav_status_file: str,
        spatial_memory_file: str,
        workspace_host: str,
        max_iterations: int,
        agent_timeout: int,
    ) -> Dict[str, Any]:
        """Build nav_status, persist files, spawn the nav_agent subprocess,
        and register the record. Called by _start_nav_loop inside a
        try/except so failures here can trigger session rollback."""
        now_iso = self._utc_now_iso()
        nav_status: Dict[str, Any] = {
            "task_id": loop_id,
            "task_type": task_type,
            "nav_mode": nav_mode,
            "has_navmesh": has_navmesh,
            "is_gaussian": session.is_gaussian,
            "goal_type": goal_type,
            "goal_description": goal_description,
            # Agent-facing goal: only for pointnav. PromptBuilder reads this
            # field and shows absolute coordinates (navmesh) or derives polar
            # signals (mapless).
            "goal_position": goal_position,
            # Evaluation-only GT: never shown to the agent. Stored in
            # nav_status for debugging/report access via get_nav_loop_status.
            "eval_goal_position": session.eval_goal,
            "has_ground_truth": session.eval_goal is not None,
            "success_distance_threshold": self._coerce_success_threshold(
                payload.get("success_distance_threshold")
            ),
            "target_object": None,
            "reference_image": reference_image,
            "substeps": [],
            "current_substep_index": 0,
            "session_id": bridge_session_id,
            "status": "in_progress",
            "nav_phase": "decomposing",
            "total_steps": 0,
            "collisions": 0,
            "current_position": None,
            "geodesic_distance": None,
            "rooms_discovered": [],
            "last_visual": None,
            "last_action": None,
            "action_history": [],
            "spatial_memory_file": spatial_memory_file,
            "finding": None,
            "error": None,
            "updated_at": now_iso,
            "state_version": 1,
            # Initial _debug snapshot so session_stats has GT metrics
            # even if the nav_loop is stopped before any update patch.
            "_debug": self._build_debug_snapshot(session),
        }
        spatial_memory: Dict[str, Any] = {
            "task_id": loop_id,
            "grid_resolution_m": 0.5,
            "snapshots": [],
            "rooms": {},
            "object_sightings": {},
        }
        # spatial_memory.json is bridge-owned and has no cross-process
        # writer, so the plain atomic persist is fine there.
        self._persist_json_atomic(spatial_memory_file, spatial_memory)
        # nav_status.json has a cross-process fallback writer in
        # tools/nav_agent.py:mark_terminal_status — MUST use the locked
        # wrapper so the two sides serialize.
        self._persist_nav_status_locked(nav_status_file, nav_status)

        nav_loop_script = self._resolve_nav_loop_script()

        env = os.environ.copy()
        # Remove SOCKS proxy — httpx requires 'socksio' for socks5 and
        # the LLM API calls work fine with http_proxy/https_proxy alone.
        env.pop("all_proxy", None)
        env.pop("ALL_PROXY", None)
        # LLM API configuration (for nav_agent.py direct mode)
        # Only propagate if actually set — avoid injecting empty strings
        # that would block nav_agent.py's own dotenv loading.
        for _key in ("NAV_LLM_API_KEY", "NAV_LLM_BASE_URL", "NAV_LLM_MODEL", "NAV_ARTIFACTS_DIR"):
            _val = os.environ.get(_key)
            if _val:
                env[_key] = _val
        # Bridge connection (agent calls bridge HTTP API directly)
        # Use adapter's actual listen address, falling back to env/defaults
        env["NAV_BRIDGE_HOST"] = getattr(self, "bridge_host", None) or os.environ.get("NAV_BRIDGE_HOST", "127.0.0.1")
        env["NAV_BRIDGE_PORT"] = str(getattr(self, "bridge_port", None) or os.environ.get("NAV_BRIDGE_PORT", "18911"))
        # Shared configuration
        env["NAV_MAX_ITERATIONS"] = str(max_iterations)
        env["NAV_AGENT_TIMEOUT"] = str(agent_timeout)
        env["NAV_WORKSPACE_HOST"] = workspace_host

        cmd = [sys.executable, nav_loop_script, nav_status_file]

        log_path = nav_status_file + ".loop.log"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )

        # From here on the subprocess is alive. If anything below raises
        # we MUST kill it before re-raising — otherwise we leave an orphan
        # process untracked by self._nav_loops, which would never be
        # cleaned up by _cleanup_finished_nav_loops or _stop_all_nav_loops.
        # The outer _start_nav_loop wrapper handles session-state rollback.
        try:
            record = _NavLoopRecord(
                loop_id=loop_id,
                process=proc,
                session_id=bridge_session_id,
                task_type=task_type,
                nav_mode=nav_mode,
                has_navmesh=has_navmesh,
                nav_status_file=nav_status_file,
                spatial_memory_file=spatial_memory_file,
                log_file=log_path,
                started_at_s=time.monotonic(),
                nav_status=copy.deepcopy(nav_status),
                state_version=1,
            )
            # Note: session.last_goal / session.eval_goal / session.mapless
            # were already committed in the pending phase of _start_nav_loop
            # under the rollback envelope. The previously-duplicated block
            # here was deleted in Phase 5 (S3 fix) — see plan.

            with self._nav_loop_lock:
                self._nav_loops[loop_id] = record
                self._register_nav_loop_policy(
                    session_id=record.session_id,
                    nav_mode=record.nav_mode,
                    loop_id=record.loop_id,
                )
        except BaseException as exc:
            # Best-effort orphan termination. Logged but never raised
            # from here so the original exception propagates.
            self._logger.error(
                "nav_loop registration failed after Popen; killing orphan "
                "subprocess pid=%s loop_id=%s: %s",
                proc.pid,
                loop_id,
                exc,
            )
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
            except Exception as kill_exc:
                self._logger.error(
                    "Failed to kill orphan subprocess pid=%s: %s",
                    proc.pid,
                    kill_exc,
                )
            # Also remove the record from _nav_loops if it managed to
            # land before the failure. This guards against a future
            # reordering where the dict insert happens before policy.
            with self._nav_loop_lock:
                self._nav_loops.pop(loop_id, None)
            raise

        self._logger.info(
            "nav_loop started loop_id=%s pid=%s nav_status_file=%s",
            loop_id,
            proc.pid,
            nav_status_file,
        )
        return {
            "loop_id": loop_id,
            "pid": proc.pid,
            "nav_status_file": nav_status_file,
            "spatial_memory_file": spatial_memory_file,
            "task_type": task_type,
            "nav_mode": nav_mode,
            "has_navmesh": has_navmesh,
            "log_file": log_path,
            "status": "started",
            "state_version": 1,
        }

    def _get_nav_loop_status(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        loop_id = payload.get("loop_id")
        if not isinstance(loop_id, str) or not loop_id:
            raise HabitatAdapterError(
                'Field "payload.loop_id" must be a non-empty str'
            )

        include_nav_status = self._coerce_bool(
            payload.get("include_nav_status", False),
            field_name="payload.include_nav_status",
        )

        self._cleanup_finished_nav_loops()
        with self._nav_loop_lock:
            record = self._nav_loops.get(loop_id) or self._closed_nav_loops.get(loop_id)
        if record is None:
            raise HabitatAdapterError(f"Unknown loop_id: {loop_id}")

        if record.returncode is None:
            polled = record.process.poll()
            if polled is not None:
                self._finalize_nav_loop_record(
                    record, int(polled), reason="process_exit"
                )
                with self._nav_loop_lock:
                    record = self._closed_nav_loops.get(loop_id, record)

        result = self._build_nav_loop_status_result(record)

        nav_status_file = record.nav_status_file

        # Refresh _debug for ACTIVE loops inside the nav_loop_lock to avoid
        # race with update_nav_loop_status (which also writes record.nav_status
        # and nav_status_file under the same lock). Closed loops keep their
        # terminal snapshot — rebuilding from current session state would
        # corrupt historical metrics if the session has since moved or
        # started another loop.
        if include_nav_status:
            with self._nav_loop_lock:
                if loop_id in self._nav_loops:
                    # Re-fetch record under lock in case it was finalized
                    record = self._nav_loops.get(loop_id, record)
                    session = self._sessions.get(record.session_id)
                    if session is not None and isinstance(record.nav_status, dict):
                        # Restore last_goal/eval_goal from nav_status if session
                        # attributes got cleared (e.g., bridge process restart).
                        if session.last_goal is None:
                            goal_pos = record.nav_status.get("goal_position")
                            if isinstance(goal_pos, list) and len(goal_pos) >= 3:
                                session.last_goal = [float(v) for v in goal_pos[:3]]
                        if session.eval_goal is None:
                            eg = record.nav_status.get("eval_goal_position")
                            if isinstance(eg, list) and len(eg) >= 3:
                                session.eval_goal = [float(v) for v in eg[:3]]
                        record.nav_status["_debug"] = self._build_debug_snapshot(session)
                        try:
                            # Locked wrapper — must serialize with the
                            # subprocess-side mark_terminal_status fallback.
                            # Codex P2 follow-up: bypassing the lock here
                            # re-opened the B1 race for status refresh.
                            self._persist_nav_status_locked(
                                nav_status_file, record.nav_status
                            )
                        except Exception:
                            pass
                # Snapshot nav_status while still holding the lock so the
                # returned copy is consistent with the just-persisted file.
                result["nav_status"] = copy.deepcopy(record.nav_status)

            parsed_file_status, file_error = self._read_nav_status_file(nav_status_file)
            if file_error is not None:
                result["nav_status_file_error"] = file_error
            elif parsed_file_status != result["nav_status"]:
                result["nav_status_file_mismatch"] = True

        return result

    def _update_nav_loop_status(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id

        loop_id = payload.get("loop_id")
        if not isinstance(loop_id, str) or not loop_id:
            raise HabitatAdapterError(
                'Field "payload.loop_id" must be a non-empty str'
            )

        patch = payload.get("patch")
        if not isinstance(patch, Mapping):
            raise HabitatAdapterError('Field "payload.patch" must be a JSON object')

        expected_version_raw = payload.get("expected_version")
        expected_version: Optional[int] = None
        if expected_version_raw is not None:
            expected_version = self._coerce_non_negative_int(
                expected_version_raw, "payload.expected_version"
            )

        self._cleanup_finished_nav_loops()
        closed_record: Optional[_NavLoopRecord] = None
        polled: Optional[int] = None
        with self._nav_loop_lock:
            record = self._nav_loops.get(loop_id)
            if record is None:
                closed_record = self._closed_nav_loops.get(loop_id)
                if closed_record is None:
                    raise HabitatAdapterError(f"Unknown loop_id: {loop_id}")
            else:
                proc_returncode = record.process.poll()
                if proc_returncode is not None:
                    polled = int(proc_returncode)

        if closed_record is not None:
            raise HabitatAdapterError(
                f'Loop "{loop_id}" is not running '
                f"(proc_status={self._nav_loop_proc_status(closed_record)}); "
                "nav-loop updates are only allowed while the loop is active"
            )

        if record is None:
            raise HabitatAdapterError(f"Unknown loop_id: {loop_id}")
        if polled is not None:
            self._finalize_nav_loop_record(record, polled, reason="process_exit")
            raise HabitatAdapterError(
                f'Loop "{loop_id}" is not running (proc_status=exited({polled})); '
                "nav-loop updates are only allowed while the loop is active"
            )

        updated_status: Dict[str, Any]
        with self._nav_loop_lock:
            record = self._nav_loops.get(loop_id)
            if record is None:
                closed_record = self._closed_nav_loops.get(loop_id)
                if closed_record is not None:
                    raise HabitatAdapterError(
                        f'Loop "{loop_id}" is not running '
                        f"(proc_status={self._nav_loop_proc_status(closed_record)}); "
                        "nav-loop updates are only allowed while the loop is active"
                    )
                raise HabitatAdapterError(f"Unknown loop_id: {loop_id}")

            current_version = record.state_version
            if expected_version is not None and expected_version != current_version:
                raise HabitatAdapterError(
                    "payload.expected_version mismatch: "
                    f"expected {expected_version}, current {current_version}"
                )

            session = self._sessions.get(record.session_id)
            sim_step = session.step_count if session is not None else None
            updated_status = self._apply_nav_status_patch(
                record.nav_status, patch, sim_step_count=sim_step
            )
            self._validate_mapless_visual_grounding_patch(
                current_nav_status=record.nav_status,
                patch=patch,
                updated_nav_status=updated_status,
            )
            next_version = current_version + 1
            updated_status["state_version"] = next_version
            updated_status["updated_at"] = self._utc_now_iso()

            # Auto-populate current_position from simulator for non-mapless
            # sessions. This ensures action_history normalization has pos data
            # even if the agent didn't explicitly include it in the patch.
            if session is not None and not session.mapless:
                gt_pos = self._to_numeric_list(self._current_position(session))
                if gt_pos is not None:
                    updated_status["current_position"] = gt_pos

            # Always inject GT debug data for developer dashboard (TUI).
            # This section is NOT visible to the agent — nav_agent.py prompt
            # never reads _debug fields, and agent tools don't expose them.
            if session is not None:
                # Ensure session.last_goal is set from nav_status for _debug
                if session.last_goal is None:
                    goal_pos = updated_status.get("goal_position")
                    if isinstance(goal_pos, list) and len(goal_pos) >= 3:
                        session.last_goal = [float(v) for v in goal_pos[:3]]
                if session.eval_goal is None:
                    eg = updated_status.get("eval_goal_position")
                    if isinstance(eg, list) and len(eg) >= 3:
                        session.eval_goal = [float(v) for v in eg[:3]]

                updated_status["_debug"] = self._build_debug_snapshot(session)

                # Persist polar goal info (relative, safe for mapless) at
                # top level so nav_agent.py can read them for the prompt.
                polar_summary = self._build_state_summary(session)
                if polar_summary.get("euclidean_distance_to_goal") is not None:
                    updated_status["euclidean_distance_to_goal"] = polar_summary["euclidean_distance_to_goal"]
                if polar_summary.get("goal_direction_deg") is not None:
                    updated_status["goal_direction_deg"] = polar_summary["goal_direction_deg"]

            # Locked wrapper — see _persist_nav_status_locked docstring.
            self._persist_nav_status_locked(
                record.nav_status_file, updated_status
            )
            record.nav_status = copy.deepcopy(updated_status)
            record.state_version = next_version
            result = self._build_nav_loop_status_result(record)

        result["nav_status"] = copy.deepcopy(updated_status)
        return result

    def _stop_nav_loop(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        loop_id = payload.get("loop_id")
        if not isinstance(loop_id, str) or not loop_id:
            raise HabitatAdapterError(
                'Field "payload.loop_id" must be a non-empty str'
            )

        with self._nav_loop_lock:
            record = self._nav_loops.get(loop_id)
        if record is None:
            with self._nav_loop_lock:
                closed_record = self._closed_nav_loops.get(loop_id)
            if closed_record is None:
                raise HabitatAdapterError(f"Unknown loop_id: {loop_id}")
            return {
                "loop_id": loop_id,
                "stopped": False,
                "already_stopped": True,
                "returncode": closed_record.returncode,
            }

        returncode = self._terminate_nav_loop(record, reason="stop_nav_loop")
        self._logger.info(
            "nav_loop stopped loop_id=%s returncode=%s", loop_id, returncode
        )
        return {
            "loop_id": loop_id,
            "stopped": True,
            "returncode": returncode,
        }

    def _get_runtime_status_action(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        include_nav_status = self._coerce_bool(
            payload.get("include_nav_status", False),
            field_name="payload.include_nav_status",
        )
        runtime = self.get_runtime_status()
        if include_nav_status:
            with self._nav_loop_lock:
                records = {
                    record.loop_id: record
                    for record in list(self._nav_loops.values())
                    + list(self._closed_nav_loops.values())
                }
            for key in ("nav_loops", "recently_closed_nav_loops"):
                for loop in runtime.get(key, []):
                    loop_id = loop.get("loop_id")
                    if not isinstance(loop_id, str):
                        continue
                    record = records.get(loop_id)
                    if record is None:
                        continue
                    loop["nav_status"] = copy.deepcopy(record.nav_status)
                    parsed_file_status, file_error = self._read_nav_status_file(
                        record.nav_status_file
                    )
                    if file_error is not None:
                        loop["nav_status_file_error"] = file_error
                    elif parsed_file_status != record.nav_status:
                        loop["nav_status_file_mismatch"] = True
        return runtime

    def _has_running_nav_loop_for_session(self, session_id: str) -> bool:
        self._cleanup_finished_nav_loops()
        with self._nav_loop_lock:
            return any(
                record.session_id == session_id for record in self._nav_loops.values()
            )

    def _assert_map_planning_allowed(self, session: _Session, action_name: str) -> None:
        self._cleanup_finished_nav_loops()
        with self._nav_loop_lock:
            mapless_loop_ids = self._session_mapless_loop_ids.get(
                session.session_id, set()
            )
        if not mapless_loop_ids:
            return
        raise HabitatAdapterError(
            f'Action "{action_name}" is forbidden while session "{session.session_id}" '
            "is in mapless nav-loop mode; use step_action/get_metrics/get_visuals instead"
        )

    def _register_nav_loop_policy(
        self, session_id: str, nav_mode: str, loop_id: str
    ) -> None:
        if nav_mode != "mapless":
            return
        loop_ids = self._session_mapless_loop_ids.setdefault(session_id, set())
        loop_ids.add(loop_id)

    def _unregister_nav_loop_policy(
        self, session_id: str, nav_mode: str, loop_id: str
    ) -> None:
        if nav_mode != "mapless":
            return
        loop_ids = self._session_mapless_loop_ids.get(session_id)
        if not loop_ids:
            return
        loop_ids.discard(loop_id)
        if not loop_ids:
            self._session_mapless_loop_ids.pop(session_id, None)
            # Clear mapless flag so session returns to normal behavior
            session = self._sessions.get(session_id)
            if session is not None:
                session.mapless = False

    def _stop_all_nav_loops(self, reason: str) -> None:
        with self._nav_loop_lock:
            loop_ids = list(self._nav_loops.keys())
        for loop_id in loop_ids:
            with self._nav_loop_lock:
                record = self._nav_loops.get(loop_id)
            if record is not None:
                self._terminate_nav_loop(record, reason=reason)

    def _stop_nav_loops_for_session(self, session_id: str, reason: str) -> None:
        with self._nav_loop_lock:
            loop_ids = [
                loop_id
                for loop_id, record in self._nav_loops.items()
                if record.session_id == session_id
            ]
        for loop_id in loop_ids:
            with self._nav_loop_lock:
                record = self._nav_loops.get(loop_id)
            if record is not None:
                self._terminate_nav_loop(record, reason=reason)

    def _terminate_nav_loop(self, record: _NavLoopRecord, reason: str) -> int:
        with self._nav_loop_lock:
            proc = record.process
            if record.returncode is not None:
                return int(record.returncode)

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        returncode = proc.returncode
        if returncode is None:
            returncode = -15
        self._finalize_nav_loop_record(record, int(returncode), reason=reason)
        return int(returncode)

    def _cleanup_finished_nav_loops(self) -> None:
        with self._nav_loop_lock:
            loop_ids = list(self._nav_loops.keys())
        for loop_id in loop_ids:
            with self._nav_loop_lock:
                record = self._nav_loops.get(loop_id)
            if record is None or record.returncode is not None:
                continue
            polled = record.process.poll()
            if polled is None:
                continue
            self._finalize_nav_loop_record(record, int(polled), reason="process_exit")

    def _finalize_nav_loop_record(
        self, record: _NavLoopRecord, returncode: int, reason: str
    ) -> None:
        parsed_file_status, _ = self._read_nav_status_file(record.nav_status_file)
        should_persist_nav_status = False
        with self._nav_loop_lock:
            if record.returncode is not None:
                return
            source_status: Mapping[str, Any]
            if parsed_file_status is not None:
                source_status = parsed_file_status
            else:
                source_status = record.nav_status

            record.nav_status = copy.deepcopy(dict(source_status))
            state_version = record.nav_status.get("state_version")
            if (
                isinstance(state_version, int)
                and not isinstance(state_version, bool)
                and state_version >= 0
            ):
                record.state_version = int(state_version)

            (
                record.nav_status,
                should_persist_nav_status,
            ) = self._force_nav_loop_terminal_status(
                record.nav_status,
                returncode=returncode,
                reason=reason,
            )
            state_version = record.nav_status.get("state_version")
            if (
                isinstance(state_version, int)
                and not isinstance(state_version, bool)
                and state_version >= 0
            ):
                record.state_version = int(state_version)
            record.returncode = int(returncode)
            record.ended_at_s = time.monotonic()
            self._nav_loops.pop(record.loop_id, None)
            self._archive_closed_nav_loop(record)
            self._unregister_nav_loop_policy(
                session_id=record.session_id,
                nav_mode=record.nav_mode,
                loop_id=record.loop_id,
            )
        if should_persist_nav_status:
            try:
                # Locked wrapper — the subprocess may still be alive and
                # attempting a mark_terminal_status fallback write at the
                # same moment we're finalizing, so serialize.
                self._persist_nav_status_locked(
                    record.nav_status_file, record.nav_status
                )
            except OSError as exc:
                self._logger.warning(
                    "Failed to persist terminal nav_status loop_id=%s: %s",
                    record.loop_id,
                    exc,
                )
        self._logger.info(
            "nav_loop finalized loop_id=%s returncode=%s reason=%s",
            record.loop_id,
            record.returncode,
            reason,
        )

    def _archive_closed_nav_loop(self, record: _NavLoopRecord) -> None:
        self._closed_nav_loops[record.loop_id] = record
        while len(self._closed_nav_loops) > _MAX_CLOSED_NAV_LOOPS:
            oldest_loop_id = next(iter(self._closed_nav_loops.keys()))
            if oldest_loop_id == record.loop_id and len(self._closed_nav_loops) == 1:
                break
            self._closed_nav_loops.pop(oldest_loop_id, None)

    @staticmethod
    def _nav_loop_proc_status(record: _NavLoopRecord) -> str:
        if record.returncode is None:
            return "running"
        if record.returncode == 0:
            return "finished"
        return f"exited({record.returncode})"

    def _build_nav_loop_status_result(self, record: _NavLoopRecord) -> Dict[str, Any]:
        return {
            "loop_id": record.loop_id,
            "pid": record.process.pid,
            "proc_status": self._nav_loop_proc_status(record),
            "session_id": record.session_id,
            "task_type": record.task_type,
            "nav_mode": record.nav_mode,
            "has_navmesh": record.has_navmesh,
            "nav_status_file": record.nav_status_file,
            "spatial_memory_file": record.spatial_memory_file,
            "log_file": record.log_file,
            "started_at_s": round(record.started_at_s, 3),
            "ended_at_s": (
                round(record.ended_at_s, 3) if record.ended_at_s is not None else None
            ),
            "returncode": record.returncode,
            "state_version": record.state_version,
        }
