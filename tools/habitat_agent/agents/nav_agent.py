"""NavAgent — autonomous navigation loop subprocess.

This is the long-running per-loop subprocess that drives the LLM
through the bridge tool API. Each invocation processes one nav loop
to terminal state (reached / blocked / timeout / error) and writes
session stats on exit.

Phase 1 PR 3 moved this class out of ``tools/nav_agent.py`` into the
``habitat_agent.agents`` package. The class body itself is unchanged
— only the imports were updated to use the new package paths instead
of the legacy ``habitat_agent_core`` shim. The CLI entry point lives
in :mod:`habitat_agent.agents.nav_agent_main`.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from analytics.session_stats import collect_session_stats
from analytics.trace_writer import append_round_event, append_trace
from habitat_agent.prompts.legacy_builder import PromptBuilder
from habitat_agent.runtime.bridge_client import BridgeClient
from habitat_agent.runtime.constants import TERMINAL_STATUSES
from habitat_agent.runtime.file_io import copy_file_atomic_under_lock
from habitat_agent.runtime.image_io import build_image_user_message
from habitat_agent.runtime.llm_client import get_openai
from habitat_agent.runtime.log import log
from habitat_agent.runtime.terminal_status import mark_terminal_status
# Phase 2 PR 4: dispatch cutover. Tool schemas and execution now go
# through ToolRegistry, which is populated by the Tool subclasses in
# habitat_agent.tools.{navigation,perception,mapping,status,session}.
# The import of `habitat_agent.tools` below triggers registration as
# a side effect — this is why the import is load-bearing, not just
# for the ToolRegistry/ToolContext/RoundState symbols.
import habitat_agent.tools  # noqa: F401  — registers all 16 tools
from habitat_agent.tools.base import (
    RoundState,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


class NavAgent:
    """Main navigation agent loop."""

    def __init__(self, status_file: str):
        self.status_file = os.path.abspath(status_file)
        self.events_file = self.status_file + ".events.jsonl"
        self.trace_file = self.status_file + ".trace.jsonl"
        self.last_good_file = self.status_file + ".last-good.json"

        # Read initial nav_status
        with open(self.status_file, "r", encoding="utf-8") as f:
            self.nav_status: Dict[str, Any] = json.load(f)

        self.loop_id = self.nav_status.get("task_id", "")
        self.session_id = self.nav_status.get("session_id", "")
        self.nav_mode = self.nav_status.get("nav_mode", "mapless")
        self.task_type = self.nav_status.get("task_type", "pointnav")

        # Config from env
        self.max_iterations = int(os.environ.get("NAV_MAX_ITERATIONS", "50"))
        self.agent_timeout = int(os.environ.get("NAV_AGENT_TIMEOUT", "120"))
        self.idle_threshold = int(os.environ.get("NAV_IDLE_ESCALATION_THRESHOLD", "3"))

        # LLM client
        openai_mod = get_openai()
        api_key = os.environ.get("NAV_LLM_API_KEY", "")
        if not api_key:
            raise RuntimeError("NAV_LLM_API_KEY env var is required")
        base_url = os.environ.get("NAV_LLM_BASE_URL", "https://api.anthropic.com/v1/")
        self.model = os.environ.get("NAV_LLM_MODEL", "claude-sonnet-4-20250514")
        self.max_tokens = int(os.environ.get("NAV_LLM_MAX_TOKENS", "4096"))
        self.client = openai_mod.OpenAI(api_key=api_key, base_url=base_url)

        # Bridge client
        bridge_host = os.environ.get("NAV_BRIDGE_HOST", "127.0.0.1")
        bridge_port = int(os.environ.get("NAV_BRIDGE_PORT", "18911"))
        self.bridge = BridgeClient(host=bridge_host, port=bridge_port)
        self.bridge.session_id = self.session_id

        # Output directory for images
        self.output_dir = os.path.dirname(self.status_file)

        # State tracking
        self.state_version_ref = [self.nav_status.get("state_version", 1)]
        self.consecutive_idle = 0
        self.prev_total_steps: Optional[int] = None
        # Counter for B3 fix: how many times _reload_nav_status has failed.
        # Surfaced in logs so operators notice cumulative read errors that
        # would otherwise be silently absorbed.
        self.reload_failures = 0

        # Prompt builder
        self.prompt_builder = PromptBuilder(self.nav_mode)

        # Phase 3: construct MemoryBundle with SpatialMemory if available.
        from habitat_agent.memory.base import MemoryBundle
        from habitat_agent.memory.spatial import SpatialMemory

        spatial_file = self.nav_status.get("spatial_memory_file", "")
        self.memory_bundle = MemoryBundle()
        if spatial_file:
            self.memory_bundle.register(SpatialMemory(spatial_file))

        # Phase 2 PR 4: Tool dispatch goes through ToolRegistry. The
        # per-round mutable state that legacy ToolExecutor stored as
        # instance attributes (captured_images, last_collided, etc.)
        # now lives on ctx.round_state. state_version_ref is still a
        # list-of-one so update_nav_status can mutate it from inside
        # the Tool and the next round's expected_version stays in sync.
        self.tool_ctx = ToolContext(
            bridge=self.bridge,
            session_id=self.session_id,
            loop_id=self.loop_id,
            output_dir=self.output_dir,
            workspace_host=os.environ.get("NAV_WORKSPACE_HOST", self.output_dir),
            nav_mode=self.nav_mode,
            task_type=self.task_type,
            is_gaussian=self.nav_status.get("is_gaussian", False),
            state_version_ref=self.state_version_ref,
            memory_bundle=self.memory_bundle,
        )

        # Tool schemas from the Registry — replaces build_tool_schemas.
        # available_for() applies both nav_mode and task_type gates,
        # so nav_agent automatically doesn't see the 5 chat-only
        # session tools.
        self.tools = ToolRegistry.build_openai_schemas(
            nav_mode=self.nav_mode,
            task_type=self.task_type,
        )

        # Conversation history (fresh per loop — the core fix)
        self.conversation: List[Dict[str, Any]] = []
        self.system_prompt = self.prompt_builder.build_system_prompt(self.nav_status)

        # Pending images from tool calls (shown at start of next round)
        self.pending_images: List[str] = []

    def _reload_nav_status(self) -> Dict[str, Any]:
        """Re-read nav_status.json from disk.

        On failure, log and increment self.reload_failures so operators
        can notice cumulative read errors. The previous silent except:pass
        meant the agent could run on stale state for many rounds without
        any signal (bug B3 in PR #28 audit).
        """
        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                self.nav_status = json.load(f)
            self.state_version_ref[0] = self.nav_status.get("state_version", 1)
        except Exception as exc:
            self.reload_failures += 1
            log(
                f"WARNING: _reload_nav_status failed (count={self.reload_failures}): "
                f"{exc!r}"
            )
        return self.nav_status

    def _is_terminal(self) -> bool:
        return self.nav_status.get("status", "") in TERMINAL_STATUSES

    def run(self) -> None:
        """Main agent loop."""
        log(f"NavAgent starting loop_id={self.loop_id} task={self.task_type} mode={self.nav_mode}")
        self._current_round = 0

        for round_idx in range(self.max_iterations):
            self._current_round = round_idx + 1
            # Check bridge health
            if not self.bridge.healthz():
                log("Bridge is not responding — exiting")
                # Bridge is known unreachable here → skip path 1, go straight
                # to the locked local fallback by passing bridge=None.
                mark_terminal_status(
                    self.status_file,
                    "error",
                    "nav_agent: bridge not reachable",
                    bridge=None,
                    loop_id=None,
                )
                break

            # Re-read nav_status
            self._reload_nav_status()

            # Check terminal
            if self._is_terminal():
                log(f"Navigation finished with status: {self.nav_status.get('status')}")
                break

            # Save last-good snapshot via locked atomic copy. Bug B5 fix:
            # the previous shutil.copy2 was a non-atomic two-phase read+write
            # that could produce a torn destination if the bridge replaced
            # status_file mid-copy. copy_file_atomic_under_lock acquires the
            # same sidecar flock as the bridge writers, then uses temp+rename.
            try:
                copy_file_atomic_under_lock(
                    self.status_file,
                    self.last_good_file,
                    lock_path=self.status_file,
                )
            except Exception as exc:
                log(f"WARNING: last_good snapshot failed: {exc}")

            append_round_event(self.events_file, "round_start", round_idx + 1, self.nav_status)

            # Build user message
            round_text = self.prompt_builder.build_round_message(
                self.nav_status, round_idx, self.consecutive_idle, self.idle_threshold,
                memory_bundle=self.memory_bundle,
            )

            # Round text goes into conversation history (persistent).
            # Pending images are ephemeral — injected into messages for
            # this API call only, not stored in conversation. The
            # helper sends ALL captured images (no [:4] cap) because
            # the agent may have chained panorama + look in the prior
            # round and we must not silently drop the newest frame.
            user_content = round_text
            pending_image_msg = None
            if self.pending_images:
                pending_image_msg = build_image_user_message(
                    "Images from your previous actions. Describe what you see in your perception field.",
                    self.pending_images,
                )
                self.pending_images = []

            self.conversation.append({"role": "user", "content": user_content})

            log(
                f"Round {round_idx + 1} | task_type={self.task_type} | mode={self.nav_mode} "
                f"| phase={self.nav_status.get('nav_phase')} | status={self.nav_status.get('status')} "
                f"| steps={self.nav_status.get('total_steps', 0)}"
            )

            # Run agentic tool loop (pass ephemeral image message if any)
            try:
                self._run_round(ephemeral_image_msg=pending_image_msg)
            except Exception as exc:
                log(f"Round {round_idx + 1} error: {exc}")
                traceback.print_exc()
                # Don't crash — try next round

            # Re-read status after round
            self._reload_nav_status()

            # Idle detection
            post_total_steps = self.nav_status.get("total_steps", 0)
            if self.prev_total_steps is not None:
                if post_total_steps == self.prev_total_steps:
                    self.consecutive_idle += 1
                else:
                    self.consecutive_idle = 0
            self.prev_total_steps = post_total_steps

            append_round_event(self.events_file, "round_end", round_idx + 1, self.nav_status)
            log(
                f"Round {round_idx + 1} done | now: {self.nav_status.get('status', '?')} "
                f"| steps: {post_total_steps} | idle_streak: {self.consecutive_idle}"
            )

            # Terminal check after round
            if self._is_terminal():
                log(f"Agent set terminal status: {self.nav_status.get('status')}")
                break
        else:
            # Max iterations reached. Bridge is presumed reachable here
            # (we've been making bridge calls every round) so route the
            # terminal status through the bridge for single-owner consistency.
            log(f"Max iterations ({self.max_iterations}) reached, setting timeout")
            mark_terminal_status(
                self.status_file,
                "timeout",
                "nav_agent max iterations exceeded",
                bridge=self.bridge,
                loop_id=self.loop_id,
            )

        collect_session_stats(
            self.status_file, self.events_file,
            bridge=self.bridge, loop_id=self.loop_id,
        )

    def _run_round(self, ephemeral_image_msg: Optional[Dict[str, Any]] = None) -> None:
        """Execute one round of LLM interaction with tool calls."""
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation
        # Inject ephemeral image (not in conversation history — one-time only)
        if ephemeral_image_msg is not None:
            messages.append(ephemeral_image_msg)

        # Retry with exponential backoff on transient errors
        max_api_retries = 3
        images_this_round: List[str] = []

        def _trace_llm(resp):
            usage = getattr(resp, "usage", None)
            append_trace(
                self.trace_file, "llm_call", self._current_round,
                model=self.model,
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )

        for attempt in range(max_api_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    max_tokens=self.max_tokens,
                    timeout=self.agent_timeout,
                )
                _trace_llm(response)
                break
            except Exception as exc:
                if attempt < max_api_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log(f"LLM API error (attempt {attempt + 1}): {exc}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        # Process response, handle tool calls in a loop
        max_tool_loops = 20  # safety limit
        for _ in range(max_tool_loops):
            choice = response.choices[0]
            finish = choice.finish_reason

            if finish == "tool_calls" and choice.message.tool_calls:
                # Record assistant message
                assistant_dict = {"role": "assistant", "content": choice.message.content}
                tool_calls_list = []
                for tc in choice.message.tool_calls:
                    tool_calls_list.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                    )
                assistant_dict["tool_calls"] = tool_calls_list
                self.conversation.append(assistant_dict)
                messages.append(assistant_dict)

                # Execute each tool call through the Registry.
                # `captured_images` on ctx.round_state accumulates across
                # every tool call in this turn; we clear it at the start
                # of each dispatch so `new_images` only reflects THIS
                # call, matching legacy ToolExecutor.execute semantics.
                pending_visual_injection: List[str] = []
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    append_trace(self.trace_file, "tool_call", self._current_round, tool=tc.function.name, args=args)

                    self.tool_ctx.round_state.captured_images = []
                    result: ToolResult = ToolRegistry.dispatch(
                        tc.function.name, args, self.tool_ctx
                    )
                    new_images = list(self.tool_ctx.round_state.captured_images)
                    images_this_round.extend(new_images)

                    # Convert ToolResult back to the JSON-text shape the
                    # LLM API expects in the tool_msg content. Match the
                    # legacy error-vs-success payloads exactly so the
                    # trace summary logic below keeps working.
                    if result.ok:
                        result_text = json.dumps(
                            result.body, ensure_ascii=False, default=str
                        )
                    else:
                        result_text = json.dumps({"error": result.error or "unknown error"})

                    # Summarize result for trace (avoid dumping full base64/JSON)
                    try:
                        robj = json.loads(result_text)
                        summary = robj.get("error") or f"ok (keys: {','.join(list(robj.keys())[:5])})"
                    except Exception:
                        summary = result_text[:120]
                    append_trace(self.trace_file, "tool_result", self._current_round, tool=tc.function.name, summary=summary)

                    tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                    self.conversation.append(tool_msg)
                    messages.append(tool_msg)

                    # Queue images from visual tools for immediate injection
                    if tc.function.name in ("look", "panorama") and new_images:
                        pending_visual_injection.extend(new_images)

                # Inject captured images for THIS API call only.
                # Images are ephemeral — the LLM writes perception text which
                # persists in action_history. No base64 in conversation history.
                # Sends ALL images (no [:4] cap) so chained tool calls
                # like panorama+look don't lose the newest frame.
                if pending_visual_injection:
                    img_msg = build_image_user_message(
                        "Here are the images from your observation. Describe what you see in your perception field.",
                        pending_visual_injection,
                    )
                    if img_msg is not None:
                        # Only add to messages (current API call), NOT to conversation history
                        messages.append(img_msg)

                # Continue with next LLM call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    max_tokens=self.max_tokens,
                    timeout=self.agent_timeout,
                )
                _trace_llm(response)
            else:
                # Final text response — record and exit
                if choice.message.content:
                    self.conversation.append({"role": "assistant", "content": choice.message.content})
                break

        # Save captured images for next round
        self.pending_images = images_this_round


__all__ = ["NavAgent"]
