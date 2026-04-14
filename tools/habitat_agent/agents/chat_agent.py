"""ChatAgent — interactive LLM + bridge tool execution + session persistence.

Hosts the chat-mode agent that powers the TUI's "Chat" view (and any
other future entrypoint that wants a stateful conversational front for
the bridge tools). Phase 1 PR 5 moved this verbatim out of
``tools/habitat_agent_tui.py`` so the legacy file can shrink to a thin
shim.

Phase 2 will likely fold this together with NavAgent under a shared
agent base class. For now we keep the implementation untouched and
just relocate it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_CHAT_SYSTEM_PROMPT = """\
You are HabitatAgent, an AI assistant that can also control a robot in a 3D indoor simulation when asked.

You can have normal conversations. Only use tools when the user explicitly asks you to \
interact with the simulation (e.g., "look around", "move forward", "start navigating", \
"initialize the scene"). Do NOT call any tool for greetings, questions, or general chat.

When the user does ask to interact with the simulation:
- Call init_scene() first if no session has been started yet.
- Use look/panorama/depth_analyze to observe, forward/turn to move.
- navigate(x,y,z) for auto-navigation, topdown() for the overhead map.
- For long tasks, launch a nav_loop sub-agent with nav_loop_start().
- After moving, briefly describe what you observe.
- Be specific about spatial details (objects, rooms, distances).

IMPORTANT — nav_loop is a background sub-agent:
- After calling nav_loop_start(), immediately reply to the user with the loop_id. Do NOT poll nav_loop_status in a loop.
- Only call nav_loop_status when the user explicitly asks for progress (e.g., "check status", "how is it going").
- The sub-agent runs independently. You do not need to wait for it or monitor it.
"""


def _build_chat_tool_schemas(nav_mode: str = "navmesh") -> List[Dict[str, Any]]:
    """Build chat-mode tool schemas from ToolRegistry.

    Phase 2 PR 4: this helper is no longer called by ChatAgent itself
    (ChatAgent.__init__ queries the Registry directly), but the
    habitat_agent_tui shim still re-exports this name for backward
    compatibility. Route it through the Registry so any external
    caller gets the same post-migration schema as ChatAgent.
    """
    import habitat_agent.tools  # noqa: F401 — populate the Registry
    from habitat_agent.tools.base import ToolRegistry

    return [
        s
        for s in ToolRegistry.build_openai_schemas(
            nav_mode=nav_mode, task_type="chat"
        )
        if s["function"]["name"] != "update_nav_status"
    ]


def _chat_sessions_dir() -> str:
    d = os.environ.get(
        "NAV_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "nav_artifacts"),
    )
    return os.path.join(d, "chat_sessions")


def _list_chat_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent chat sessions sorted by mtime descending."""
    d = _chat_sessions_dir()
    if not os.path.isdir(d):
        return []
    sessions: List[Dict[str, Any]] = []
    for name in os.listdir(d):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(d, name)
        try:
            mtime = os.path.getmtime(path)
            # Read first line for metadata
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
            meta = json.loads(first) if first else {}
            with open(path, "r", encoding="utf-8") as _fc:
                msg_count = sum(1 for _ in _fc) - 1  # exclude meta line
        except Exception:
            continue
        sessions.append({
            "session_id": meta.get("session_id", name.replace(".jsonl", "")),
            "created_at": meta.get("created_at", ""),
            "model": meta.get("model", ""),
            "messages": max(0, msg_count),
            "mtime": mtime,
            "file": path,
        })
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions[:limit]


class ChatAgent:
    """Chat-mode agent: LLM conversation + bridge tool execution + session persistence."""

    def __init__(self, bridge_url: str, resume_file: str = ""):
        # Phase 2 PR 4: dispatch moved to ToolRegistry. Importing
        # `habitat_agent.tools` populates the Registry with all 16
        # tools as a side effect, which is why this import is
        # load-bearing (not just for the symbols).
        from habitat_agent.runtime.bridge_client import BridgeClient
        from habitat_agent.runtime.llm_client import get_openai
        import habitat_agent.tools  # noqa: F401  — triggers registration
        from habitat_agent.tools.base import (
            ToolContext,
            ToolRegistry,
            ToolResult,
        )
        self._ToolRegistry = ToolRegistry
        self._ToolResult = ToolResult

        self.bridge = BridgeClient()
        self.bridge.base_url = bridge_url
        self.model = os.environ.get("NAV_LLM_MODEL", "gpt-4o")
        self._api_key = os.environ.get("NAV_LLM_API_KEY", "")
        self._base_url = os.environ.get("NAV_LLM_BASE_URL", "")
        self._openai = get_openai()
        self._client: Any = None

        artifacts_dir = os.environ.get(
            "NAV_ARTIFACTS_DIR",
            str(Path(__file__).resolve().parents[3] / "data" / "nav_artifacts"),
        )

        # ToolContext wraps bridge + per-round state. task_type="chat"
        # makes the 5 session tools visible; `update_nav_status` is
        # unconditionally not registered for chat (no loop to update).
        # nav_mode="navmesh" is the chat default; init_scene can
        # swap it if bridge reports a mapless-only scene.
        # Phase 3: empty MemoryBundle for chat mode. Chat doesn't use
        # spatial memory (no nav_status context), but the placeholder
        # allows Phase 4+ to add session-scoped memories later.
        from habitat_agent.memory.base import MemoryBundle
        self.tool_ctx = ToolContext(
            bridge=self.bridge,
            session_id="",
            loop_id="",
            output_dir=artifacts_dir,
            nav_mode="navmesh",
            task_type="chat",
            memory_bundle=MemoryBundle(),
        )

        # Tool schemas from Registry. Filter out update_nav_status —
        # it is nav_agent business and would only confuse chat users.
        # (The legacy `_build_chat_tool_schemas` did the same filter.)
        self.tools = [
            s
            for s in ToolRegistry.build_openai_schemas(
                nav_mode="navmesh", task_type="chat"
            )
            if s["function"]["name"] != "update_nav_status"
        ]

        self.conversation: List[Dict[str, Any]] = []
        self.last_response: str = ""
        self.is_gaussian: bool = False

        # Session persistence
        if resume_file and os.path.isfile(resume_file):
            self.session_file = resume_file
            self._load_session()
        else:
            import uuid as _uuid
            sid = f"chat_{int(time.time())}_{_uuid.uuid4().hex[:8]}"
            sess_dir = _chat_sessions_dir()
            os.makedirs(sess_dir, exist_ok=True)
            self.session_file = os.path.join(sess_dir, f"{sid}.jsonl")
            self.session_id = sid
            self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "type": "meta",
            "session_id": self.session_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": self.model,
        }
        with open(self.session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def _append_message(self, msg: Dict[str, Any]) -> None:
        """Append a single message to the session file."""
        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _load_session(self) -> None:
        """Load conversation from session file."""
        self.conversation = []
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("type") == "meta":
                        self.session_id = obj.get("session_id", "")
                        continue
                    if "role" in obj:
                        self.conversation.append(obj)
        except Exception:
            self.session_id = getattr(self, "session_id", "unknown")

    @property
    def client(self) -> Any:
        if self._client is None:
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = self._openai.OpenAI(**kwargs)
        return self._client

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _trim_conversation(self, max_messages: int = 80) -> None:
        """Keep conversation within bounds. No base64 images to worry about
        since images are ephemeral (injected per-call, not stored)."""
        if len(self.conversation) <= max_messages:
            return
        keep_recent = max_messages // 2
        self.conversation = self.conversation[:4] + self.conversation[-keep_recent:]

    def process_message(self, user_msg: str):
        """Generator yielding streaming events.

        Events: ("token", text), ("message_done",),
                ("tool_call", name, args), ("tool_result", name, text, images),
                ("error", text)
        """
        user_msg_obj = {"role": "user", "content": user_msg}
        self.conversation.append(user_msg_obj)
        self._append_message(user_msg_obj)
        _ephemeral_imgs: Optional[Dict[str, Any]] = None

        while True:
            try:
                msgs = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}] + self.conversation
                # Append ephemeral image (one-time, not in conversation history)
                if _ephemeral_imgs is not None:
                    msgs.append(_ephemeral_imgs)
                    _ephemeral_imgs = None  # consume once
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    tools=self.tools,
                    stream=True,
                )
            except Exception as exc:
                yield ("error", f"LLM request failed: {exc}")
                return

            # Accumulate streamed response
            content_buf = ""
            tc_map: Dict[int, Dict[str, str]] = {}  # index -> {id, name, arguments}
            finish = None

            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                delta = choice.delta
                if delta and delta.content:
                    content_buf += delta.content
                    yield ("token", delta.content)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_map:
                            tc_map[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tc_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tc_map[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tc_map[idx]["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason

            # Store assistant message in conversation
            assistant: Dict[str, Any] = {"role": "assistant", "content": content_buf or None}
            if tc_map:
                assistant["tool_calls"] = [
                    {"id": tc_map[i]["id"], "type": "function",
                     "function": {"name": tc_map[i]["name"], "arguments": tc_map[i]["arguments"]}}
                    for i in sorted(tc_map)
                ]
            self.conversation.append(assistant)
            self._append_message(assistant)

            if finish == "tool_calls" and tc_map:
                if content_buf:
                    yield ("message_done",)
                # Execute tool calls through the Registry. Clear
                # captured_images on ctx.round_state before each
                # dispatch so the per-call `images` list only
                # contains what THIS call captured (legacy parity).
                # The session tools (init_scene, nav_loop_*) may
                # mutate ctx.bridge.session_id / ctx.session_id /
                # ctx.is_gaussian; we mirror those onto self.is_gaussian
                # after each call so downstream UI reads still work.
                tool_msgs: List[Dict[str, Any]] = []
                all_images: List[str] = []
                for idx in sorted(tc_map):
                    tc_info = tc_map[idx]
                    name = tc_info["name"]
                    try:
                        args = json.loads(tc_info["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    yield ("tool_call", name, args)

                    self.tool_ctx.round_state.captured_images = []
                    result = self._ToolRegistry.dispatch(
                        name, args, self.tool_ctx
                    )
                    if result.ok:
                        result_text = json.dumps(
                            result.body, ensure_ascii=False, default=str
                        )
                    else:
                        result_text = json.dumps(
                            {"error": result.error or "unknown error"}
                        )
                    images = list(self.tool_ctx.round_state.captured_images)
                    all_images.extend(images)
                    # InitSceneTool mutates ctx.is_gaussian; mirror it
                    # onto self so the existing chat-frontend reads
                    # (e.g. banner rendering) still work.
                    self.is_gaussian = self.tool_ctx.is_gaussian

                    yield ("tool_result", name, result_text, images)
                    tool_msg = {"role": "tool", "tool_call_id": tc_info["id"], "content": result_text}
                    tool_msgs.append(tool_msg)
                    self._append_message(tool_msg)
                self.conversation.extend(tool_msgs)
                # Build ephemeral image message for next API call only.
                # Images are one-time: LLM sees them, writes perception text,
                # then images are discarded. No base64 in conversation history.
                # Sends ALL captured images (no [:4] cap) so chained
                # tool calls like panorama+look don't lose the newest
                # frame — fix for round 6 codex review.
                from habitat_agent.runtime.image_io import build_image_user_message
                _ephemeral_imgs: Optional[Dict[str, Any]] = build_image_user_message(
                    "Here are the captured images:", all_images
                )
                continue  # next LLM call
            else:
                if content_buf:
                    self.last_response = content_buf
                    yield ("message_done",)
                self._trim_conversation()
                break

    # Phase 2 PR 4: _dispatch_tool + the 5 _tool_* methods + _resolve_loop_id
    # were deleted. All tool execution now goes through
    # `ToolRegistry.dispatch(name, args, self.tool_ctx)` in process_message,
    # and the session tools (init_scene / close_session / nav_loop_*) live
    # as proper Tool subclasses in habitat_agent.tools.session. The
    # session-file persistence (write_meta / append_message / _load_session)
    # stays on ChatAgent because it is chat-UX business, not tool behaviour.



__all__ = [
    "_CHAT_SYSTEM_PROMPT",
    "_build_chat_tool_schemas",
    "_chat_sessions_dir",
    "_list_chat_sessions",
    "ChatAgent",
]
