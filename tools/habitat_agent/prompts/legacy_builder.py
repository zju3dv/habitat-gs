"""Legacy `PromptBuilder` — system + per-round prompt assembly.

Hosts the original prompt builder. New prompt logic should go in
``habitat_agent/prompts/`` (loader, fragments, controllers) using the
versioned ``PromptSpec`` / ``PromptLibrary`` system.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Builds system prompt and per-round user messages from nav_status."""

    def __init__(self, nav_mode: str):
        self.nav_mode = nav_mode
        # The .txt controller templates live next to the legacy nav_loop_prompts/
        # directory at tools/nav_loop_prompts/. From this file
        # (tools/habitat_agent/prompts/legacy_builder.py), the prompt dir is
        # 3 levels up + nav_loop_prompts/.
        self.prompt_dir = Path(__file__).resolve().parents[2] / "nav_loop_prompts"

    def _library_root(self) -> Path:
        """Path to the versioned prompt assets (fragments/ + controllers/).

        From this file (tools/habitat_agent/prompts/legacy_builder.py)
        the prompts package root is the directory containing this file.
        """
        return Path(__file__).resolve().parent

    def build_system_prompt(self, nav_status: Dict[str, Any]) -> str:
        """Build the static system prompt with navigation rules.

        Tries PromptLibrary first (YAML assets), falls
        back to the legacy hardcoded f-string if the library raises
        (e.g. missing asset files). The fallback ensures backward
        compat during the transition and in test environments that
        don't have the asset files on disk.
        """
        task_type = nav_status.get("task_type", "pointnav")

        try:
            from habitat_agent.prompts.spec import PromptLibrary
            lib = PromptLibrary(self._library_root())
            spec = lib.load_controller(task_type, self.nav_mode)
            goal_desc = nav_status.get("goal_description", "")
            substitutions = {
                "task_id": nav_status.get("task_id", ""),
                "goal_desc": goal_desc,
                "goal_description": goal_desc,
                "reference_image": nav_status.get("reference_image", "") or "",
                "task_type": task_type,
                "nav_mode": self.nav_mode,
            }
            return lib.render(spec, substitutions, self.nav_mode)
        except Exception:
            pass  # fall through to legacy path

        # ── Legacy path (fallback) ───────────────────────────────────────
        # Try to load a controller template from the old .txt files.
        # If those are also gone, produce an
        # inline fallback that preserves Rules 2-4 at minimum so the
        # agent is not left with a completely incomplete prompt.
        controller = self._load_controller(task_type, nav_status=nav_status)
        if controller.startswith("No specific controller"):
            # .txt files gone; synthesise a generic-but-functional controller
            goal_desc = nav_status.get("goal_description", "continue navigation")
            controller = (
                f"**Rule 2 — Decompose first, then execute.**\n"
                f"Analyze the current situation and create a substeps plan. "
                f"Task type: {task_type}. Navigation mode: {self.nav_mode}.\n"
                f"Goal: {goal_desc}\n"
                f"Decompose into whatever steps make sense, then execute immediately.\n\n"
                f"**Rule 3 — TaskRouter: {task_type} + {self.nav_mode}.**\n"
                f"{'You have NO access to map tools. Use forward, turn, look, panorama, depth_analyze.' if self.nav_mode == 'mapless' else 'You have full map tools: find_path, navigate, topdown, sample_point.'}\n\n"
                f"**Rule 4 — Controller.**\n"
                f"Observe the environment, plan a route, execute movements, and report when done."
            )

        system = f"""You are a navigation agent controlling a robot in a 3D indoor environment.
You navigate by calling tools (forward, turn, look, panorama, depth_analyze, etc.) and updating navigation status via update_nav_status.

## Key Rules

**Rule 1 — Tool-based interaction.**
All actions are performed through tool calls. Do NOT output shell commands.
Use the provided tools: forward, turn, look, panorama, depth_analyze, update_nav_status, export_video.
{"In navmesh mode you also have: navigate, find_path, sample_point, topdown." if self.nav_mode != "mapless" else "You are in MAPLESS mode — you have NO access to navigate, find_path, sample_point, or topdown tools. Use only forward, turn, look, panorama, depth_analyze."}

**Rule 1.5 — Movement magnitude is YOUR decision.**
The bridge's atomic step sizes are:
  - forward: **0.25m per atomic step**
  - turn: **10° per atomic step**
You may pass any positive multiple of the atomic step as `distance_m` / `degrees`. The bridge auto-decomposes a compound move into atomic steps and executes them sequentially; forward movement stops early on collision. Concrete examples:
  - `forward(distance_m=2.0)`             → 8 atomic steps, ends early if collided
  - `turn(direction="left", degrees=30)`  → 3 atomic steps

Choose the magnitude every time based on what your most recent observation actually shows — a stride that matches the visible clearance, a rotation that matches the angle you need to face. There is no default value you should fall back on. ALWAYS specify `distance_m` and `degrees` explicitly in every call, and tie the chosen value to something concrete in your perception or analysis field.

{controller}

**Rule 5 — Observe before move, then keep the round alive.**
Use a look-plan-execute cycle.{"In mapless navigating rounds, the first movement batch must be preceded by one look + image reasoning step." if self.nav_mode == "mapless" else ""}
Budget max 3 image calls per round. Every response must contain at least one tool call until the task reaches a terminal state.
Images captured by look/panorama will be shown to you at the start of the NEXT round.

**Rule 5.5 — Structured reasoning in action_history.**
Every action_history_append entry in update_nav_status MUST include three reasoning fields:
  - perception: what you actually see in the image/depth data (objects, rooms, doorways, obstacles, distances). Be specific and factual.
  - analysis: your interpretation — what does this mean for the task? Is the target visible? Which direction is promising?
  - decision: what you decided to do and WHY. State the chosen action and the reasoning.
Example:
  {{"perception": "Office room with desk and monitor on left. Open doorway ahead-right leading to hallway. Depth shows 3.2m clearance ahead.",
   "analysis": "Target is a kitchen. This looks like an office. The doorway ahead-right likely leads to a hallway connecting to other rooms.",
   "decision": "Turn right 45° to face the doorway, then move forward to enter the hallway and search for the kitchen."}}
Do NOT use one-line summaries. Each field should be 1-3 sentences with specific details.

**Rule 5.6 — Spatial memory accumulation.**
After each look/panorama, include spatial_memory_append in your update_nav_status call:
  {{"heading_deg": N, "scene_description": "brief summary", "room_label": "room name or unknown", "objects_detected": ["obj1", "obj2"]}}

**Rule 6 — State updates via update_nav_status tool.**
Use the update_nav_status tool with a patch containing ONLY mutable fields:
  substeps, current_substep_index, status, nav_phase, total_steps, collisions,
  last_action, action_history_append, spatial_memory_append, finding, error, capability_request
{"For mapless updates: include non-empty last_action, at least one action_history_append with perception/analysis/decision, and one spatial_memory_append." if self.nav_mode == "mapless" else ""}

**Rule 7 — Collision and stuck recovery.**
Check collided field in every movement response. On collision:
  - 1st collision: call depth_analyze (or look) to find which side has more clearance, turn away from the blocked side, then retry forward with a stride that matches the visible clear distance (smaller than the one that just collided).
  - 3+ consecutive collisions: rotate further (likely a larger turn than the last attempt) and attempt to escape along a different heading. The exact angle should be whatever your depth / visual data suggests is actually clear.
  - 5+ collisions without progress: set status=blocked with finding explaining the obstacle.
In every recovery action, the rotation and stride values you pick should come from what the latest perception tells you is clear — do not use fixed "recovery" constants.
Stall detection: euclidean_distance_to_goal is a STRAIGHT-LINE distance, NOT walkable distance — it may temporarily increase when you need to detour around walls or through doorways. Do NOT abandon a route just because the straight-line distance increases. Judge stalling by lack of VISUAL progress (same area for many rounds, no new observations) plus repeated collisions, not by distance alone.
In mapless mode, COLLIDED is allowed as a system safety signal even when map tools are forbidden.
CRITICAL: After ANY collision, execute at least one recovery movement in the SAME round.

**Rule 8 — Terminal handling.**
When navigation is complete (reached or blocked):
  1. Call export_video
  2. Call update_nav_status with terminal status and finding
"""
        return system

    def build_round_message(
        self,
        nav_status: Dict[str, Any],
        round_idx: int,
        consecutive_idle: int,
        idle_threshold: int,
        memory_bundle: Optional[Any] = None,
    ) -> str:
        """Build the user message for a specific round."""
        task_type = nav_status.get("task_type", "pointnav")
        nav_mode = nav_status.get("nav_mode", "mapless")
        status = nav_status.get("status", "in_progress")
        nav_phase = nav_status.get("nav_phase", "decomposing")
        total_steps = nav_status.get("total_steps", 0)
        last_action = nav_status.get("last_action", "none")
        state_version = nav_status.get("state_version", 1)
        session_id = nav_status.get("session_id", "unknown")
        loop_id = nav_status.get("task_id", "")
        goal_desc = nav_status.get("goal_description", "continue navigation")
        has_navmesh = nav_status.get("has_navmesh", False)
        substeps = nav_status.get("substeps", [])
        current_substep_index = nav_status.get("current_substep_index", 0)

        # Goal section
        if nav_mode == "mapless":
            dist = nav_status.get("euclidean_distance_to_goal")
            direction = nav_status.get("goal_direction_deg")
            if dist is not None and direction is not None:
                polar_info = f"Euclidean (straight-line) distance to goal: {dist}m | Direction: {direction}° (0=ahead, positive=turn_right to face goal, negative=turn_left to face goal). NOTE: straight-line distance can increase when detouring around obstacles — trust visual progress."
            else:
                polar_info = "Distance and direction will appear after first movement."
            goal_section = f"Goal: {goal_desc} (mapless — no absolute coordinates)\n{polar_info}"
        else:
            goal_type = nav_status.get("goal_type", "instruction")
            goal_pos = nav_status.get("goal_position")
            goal_section = f"Goal type: {goal_type}\nGoal: {goal_desc}"
            if goal_pos and isinstance(goal_pos, list) and len(goal_pos) >= 3:
                goal_section += f"\nGoal position: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]"

        # Reference image (for ImageNav)
        ref_img = nav_status.get("reference_image")
        if ref_img:
            goal_section += f"\nReference image: {ref_img}"

        # Action history
        action_history = nav_status.get("action_history", [])
        history_text = self._format_action_history(action_history[-50:], nav_mode)

        # Spatial memory summary — use MemoryBundle if provided,
        # otherwise fall back to the inline _format_spatial_summary.
        if memory_bundle is not None:
            spatial_summary = memory_bundle.render_context({"spatial": 500})
            if not spatial_summary:
                spatial_summary = "No spatial memory available."
        else:
            spatial_summary = self._format_spatial_summary(nav_status)

        # Current substep
        if substeps:
            current_substep = substeps[min(current_substep_index, len(substeps) - 1)] if substeps else {}
        else:
            current_substep = {"action": "decompose_task", "instruction": "create substeps before moving"}

        # Collisions count
        collisions = nav_status.get("collisions", 0)

        msg = f"""Round {round_idx + 1} — Continue navigation.

Session ID: {session_id}
Loop ID: {loop_id}
Task type: {task_type}
Navigation mode: {nav_mode}
Navmesh available: {has_navmesh}
{goal_section}
Status: {status}
Collisions: {collisions}
Phase: {nav_phase}
Steps so far: {total_steps}
Last action: {last_action}
Current substep: {json.dumps(current_substep, ensure_ascii=False)}
State version: {state_version}

## Spatial Memory Summary
{spatial_summary}

## Movement history (most recent last)
{history_text}
"""
        # Idle escalation
        if consecutive_idle >= idle_threshold:
            msg += f"""
## IDLE ESCALATION ({consecutive_idle} consecutive rounds with no movement)
You have NOT increased total_steps for {consecutive_idle} rounds.
CRITICAL — you MUST take action this round:
  1. If substeps exist and nav_phase is navigating/approaching: EXECUTE a movement command immediately.
  2. If stuck in decomposing: finalize substeps NOW and set nav_phase=navigating.
  3. If goal is unreachable: set status=blocked with a clear finding.
Do NOT spend this round only reading state or planning.
"""
        return msg

    def _load_controller(self, task_type: str, nav_status: Optional[Dict[str, Any]] = None) -> str:
        """Load controller template from nav_loop_prompts/*.txt.

        Templates use ${var} placeholders that are substituted from nav_status.
        """
        candidates = [
            self.prompt_dir / f"{task_type}_{self.nav_mode}.txt",
            self.prompt_dir / f"{task_type}_navmesh.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
                if nav_status:
                    goal_desc = nav_status.get("goal_description", "")
                    substitutions = {
                        "task_id": nav_status.get("task_id", ""),
                        "goal_desc": goal_desc,
                        "goal_description": goal_desc,
                        "reference_image": nav_status.get("reference_image", ""),
                        "task_type": task_type,
                        "nav_mode": self.nav_mode,
                    }
                    for var, val in substitutions.items():
                        text = text.replace(f"${{{var}}}", str(val or ""))
                return text
        return f"No specific controller for {task_type} + {self.nav_mode}. Use best judgment."

    def _format_action_history(self, entries: List[Dict], nav_mode: str) -> str:
        if not entries:
            return "  (none yet)"
        lines = []
        for i, e in enumerate(entries):
            step = e.get("step", "?")
            action = e.get("action", "?")
            collided = "YES" if e.get("collided") else "no"
            if nav_mode == "mapless":
                line = f"  [{i + 1}] step={step} {action} col={collided}"
            else:
                pos = e.get("pos", [0, 0, 0])
                if isinstance(pos, list) and len(pos) >= 3:
                    pos_str = f"[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]"
                else:
                    pos_str = "[-,-,-]"
                line = f"  [{i + 1}] step={step} {action} pos={pos_str} col={collided}"
            # Structured reasoning
            perception = e.get("perception")
            analysis = e.get("analysis")
            decision = e.get("decision")
            saw = e.get("saw")
            if perception:
                line += f"\n      perception: {perception}"
            if analysis:
                line += f"\n      analysis: {analysis}"
            if decision:
                line += f"\n      decision: {decision}"
            if not (perception or analysis or decision) and saw:
                line += f"\n      saw: {saw}"
            lines.append(line)
        return "\n".join(lines)

    def _format_spatial_summary(self, nav_status: Dict[str, Any]) -> str:
        sm_file = nav_status.get("spatial_memory_file", "")
        if not sm_file or not os.path.isfile(sm_file):
            return "No spatial memory file available."
        try:
            with open(sm_file, "r", encoding="utf-8") as f:
                sm = json.load(f)
            snapshots = len(sm.get("snapshots", []))
            rooms = list(sm.get("rooms", {}).keys())
            objects = list(sm.get("object_sightings", {}).keys())
            return (
                f"Snapshots: {snapshots}"
                f" | Rooms: {', '.join(rooms) if rooms else 'none'}"
                f" | Objects: {', '.join(objects) if objects else 'none'}"
                f"\nMANDATORY: After each look + image analysis, include spatial_memory_append in your update_nav_status call."
            )
        except Exception:
            return "Spatial memory file exists but could not be parsed."


__all__ = ["PromptBuilder"]
