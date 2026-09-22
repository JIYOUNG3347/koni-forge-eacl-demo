"""
KONI-Forge AgentBase — tool-use agent base on the OpenAI chat-completions API

All agents inherit from this class.
- OpenAI chat-completions calls with tool use
- Reasoning process streaming
- State management and logging
- Tool definition and execution
"""

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from server.core.agent_runtime_texts import TOOL_LABEL_KEYS
from server.core.config import STORAGE_ROOT
from server.core.confirm_signal import is_confirmation_question
from server.core.pending_execution import should_block as _should_block

logger = logging.getLogger("agents.foundation")

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(STORAGE_ROOT / "chroma")))

#: Tools that start real work. In guided mode they wait for the user's
#: confirmation, and they are never auto-advanced into.
EXECUTION_TOOLS = frozenset({"start_training_job", "download_model"})

GPT_MODEL = "gpt-4o"


class AgentPhase(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class ChainSignal:
    """Structured signal emitted by tool results to trigger pipeline chaining.

    Agents embed this in tool return JSON under "__chain_signal__" key.
    The dispatcher reads it to decide the next pipeline step.
    """

    stage_completed: str  # "retrieval" | "kbd" | "tuning"
    metadata: Dict[str, Any] = field(default_factory=dict)  # e.g. {"recommendation": "Fine-Tuning"}


@dataclass
class AgentEvent:
    """Message unit that agent streams to the frontend"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = "assistant"  # "user" | "assistant" | "system" | "tool_result"
    content: str = ""
    status: AgentPhase = AgentPhase.IDLE
    agent_name: str = ""
    thinking: Optional[str] = None  # reasoning process (extended thinking)
    tool_name: Optional[str] = None  # when calling a tool
    tool_input: Optional[Dict] = None
    tool_result: Optional[str] = None
    chain_signal: Optional[ChainSignal] = None  # structured chaining signal
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "status": self.status.value,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
        }
        if self.thinking:
            d["thinking"] = self.thinking
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_input is not None:
            d["tool_input"] = self.tool_input
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        return d


class AgentBase(ABC):
    """
    Base class for all agents.

    Subclasses must implement:
      - name: agent name
      - system_prompt: agent role/instruction prompt
      - tools: tool definition list passed to the model
      - execute_tool(): return tool call result
    """

    # Override in subclasses
    name: str = "base"
    description: str = "Base agent"
    system_prompt: str = "You are a helpful assistant."
    model: str = GPT_MODEL
    max_tokens: int = 4096
    default_autonomy: Optional[str] = None  # subclasses may override
    _llm_max_retries: Optional[int] = None
    _llm_timeout: Optional[Any] = None  # httpx.Timeout, typed Any to avoid a top-level import

    def __init__(self):
        self.status = AgentPhase.IDLE
        self.conversation: List[Dict[str, Any]] = []
        self._client: Optional[Any] = None
        self._autonomy: Optional[str] = None  # "auto" | "guided"
        self._autonomy_override: Optional[str] = None  # set by the dispatcher at runtime
        self._storage_root: Path = STORAGE_ROOT / "outputs"
        self._agent_provider: str = "openai"
        self._user_id: str = "default"  # owning user
        self._pipeline_source_folder: Optional[str] = None
        self._pipeline_train_job: Optional[str] = None

    def set_storage_root(self, root: Path):
        """Set the per-user storage path."""
        self._storage_root = Path(root)
        self._storage_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, *parts) -> Path:
        """Resolve a path. Everything lives under the per-user _storage_root."""
        return self._storage_root.joinpath(*parts)

    def set_autonomy(self, mode: str):
        """Let the dispatcher force an autonomy mode at runtime (auto pipelines)."""
        self._autonomy_override = mode

    def clear_autonomy_override(self):
        """Drop the runtime override and restore the configured value."""
        self._autonomy_override = None

    @property
    def autonomy(self) -> str:
        """Returns the current agent intervention level setting.
        Order: set_autonomy() override, per-user setting, global setting,
        default_autonomy, then "guided". Never cached, so a change applies at once.
        """
        # 1. Runtime override.
        if self._autonomy_override:
            return self._autonomy_override
        # 2. Per-user setting.
        user_mode = self._load_user_autonomy()
        if user_mode:
            return user_mode
        # 3. Global setting.
        config = self._load_system_config()
        global_mode = config.get("agent_autonomy")
        if global_mode:
            return global_mode
        # 4. Per-agent default.
        if self.default_autonomy:
            return self.default_autonomy
        # 5. Final fallback.
        return "guided"

    def _load_user_autonomy(self) -> Optional[str]:
        """Per-user autonomy setting, or None."""
        try:
            user_config_path = self._storage_root / "user_config.json"
            if user_config_path.exists():
                data = json.loads(user_config_path.read_text(encoding="utf-8"))
                return data.get("agent_autonomy")
        except Exception:
            pass
        return None

    @property
    def _lang(self) -> str:
        """Execution language. Falls back to English when the config cannot be read."""
        from server.core.lang import resolve_lang

        try:
            config = self._load_system_config()
            config.update(self._load_user_config())
            return resolve_lang(config=config)
        except Exception:  # noqa: BLE001 — a config read failure must not block the agent
            return "en"

    def _rt(self, name: str, **fields: object) -> str:
        """These strings are written by the code, not the LLM — changing the system
        prompt alone leaves them behind. The loop-breaking message in particular
        becomes an assistant message and therefore part of the next turn's LLM
        input, so it is behaviour, not just display.

        """
        from server.core.agent_runtime_texts import text

        return text(name, self._lang, **fields)

    def _effective_tools(self) -> List[Dict[str, Any]]:
        """Tool schemas as consumed by the model call."""
        return self.tools

    def _effective_system_prompt(self) -> str:
        """System prompt plus the autonomy instruction (single assembly point)."""
        return self.system_prompt + self.get_autonomy_instruction()

    def get_autonomy_instruction(self) -> str:
        """Instruction appended to the system prompt according to the autonomy level."""
        if self.autonomy == "auto":
            return (
                "\n\n[Agent mode: auto]\n"
                "Do not ask the user for confirmation and do not list options. "
                "Never ask questions such as 'Shall I proceed?' or 'Which model?'. "
                "Make every decision yourself and call the appropriate tool right away. "
                "If a model must be chosen, pick the largest local model that fits the GPU VRAM. "
                "Ask for confirmation only before destructive actions (deleting, large-scale changes)."
            )
        return (
            "\n\n[Agent mode: recommend, then confirm]\n"
            "For actions that run something (starting training, downloading a model), present a "
            "recommendation first and ask 'Shall I proceed?' before calling the tool. "
            "Read-only tools (listing models or datasets, GPU status, KBD probes, log analysis) can be "
            "called without confirmation; finish every step of an analysis before reporting. "
            "If the message is a greeting, thanks, small talk or a general question unrelated to the "
            "data or training pipeline, reply in plain text without calling any tool."
        )

    @property
    def client(self):
        """Lazy-init OpenAI client."""
        if self._client is None:
            retry_kwargs: Dict[str, Any] = {}
            if self._llm_max_retries is not None:
                retry_kwargs["max_retries"] = self._llm_max_retries
            if self._llm_timeout is not None:
                retry_kwargs["timeout"] = self._llm_timeout
            try:
                from openai import OpenAI as _OpenAI
            except ImportError:
                raise RuntimeError("The openai package is not installed.")
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = _OpenAI(api_key=api_key, **retry_kwargs)
            self._agent_provider = "openai"
        return self._client

    @staticmethod
    def _load_system_config() -> dict:
        import json

        config_path = STORAGE_ROOT / "system_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _load_user_config(self) -> dict:
        """Load the per-user user_config.json and filter spectra_tasks to the allowlist."""
        import json

        try:
            user_cfg_path = self._storage_root / "user_config.json"
            if not user_cfg_path.exists():
                return {}
            data = json.loads(user_cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        return data

    def _load_merged_config(self) -> dict:
        """Merge system_config with user_config, user_config winning.

        Ghost entries are filtered in _load_user_config, so after the merge
        spectra_tasks only ever holds allowlisted ids.

        """
        config = self._load_system_config()
        config.update(self._load_user_config())
        return config

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Tools available to the agent (JSON-schema tool specs, converted to OpenAI function format)."""
        return []

    @abstractmethod
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return tool call result"""
        ...

    def reset(self):
        """Reset conversation history"""
        self.conversation = []
        self.status = AgentPhase.IDLE
        self._autonomy_override = None  # drop the override
        self._client = None  # Reflect network mode changes

    async def run(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Receives user message and streams agent response.
        Yields messages in order: Reasoning -> Tool Use -> Response.

        Agentic loop: if tool_use exists, executes tool and passes result
        until agent generates final response.
        """
        # ── A pending execution tool runs as soon as the user confirms ──
        _confirm_words = {"yes", "y", "ok", "okay", "go", "proceed", "start", "sure"}
        _msg_lower = user_message.strip().lower().rstrip(".!?")
        if (
            hasattr(self, "_pending_execution")
            and self._pending_execution
            and any(w in _msg_lower for w in _confirm_words)
        ):
            pending = self._pending_execution
            self._pending_execution = None
            self._execution_confirmed = True  # allow execution for this run
            self._execution_params_seen = True

            # A model name in the user message overrides the pending input.
            import re as _re_confirm

            _model_match = _re_confirm.search(r"([\w\-\.]+:\d+\w*)", user_message)
            if _model_match:
                _selected_model = _model_match.group(1)
                _inp = pending.get("tool_input", {})
                if "model_name" in _inp:
                    _inp["model_name"] = _selected_model
            self.conversation.append({"role": "user", "content": user_message})
            self.status = AgentPhase.TOOL_USE

            yield AgentEvent(
                role="system",
                content=self._rt("analyzing"),
                status=AgentPhase.THINKING,
                agent_name=self.name,
            )
            yield AgentEvent(
                role="assistant",
                content=self._rt("tool_call", tool=pending["tool_name"]),
                status=AgentPhase.TOOL_USE,
                agent_name=self.name,
                tool_name=pending["tool_name"],
                tool_input=pending["tool_input"],
            )

            try:
                tool_result = await self.execute_tool(pending["tool_name"], pending["tool_input"])
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"

            # Parse the chain signal.
            parsed_chain_signal = None
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, dict) and "__chain_signal__" in parsed:
                    sig = parsed["__chain_signal__"]
                    parsed_chain_signal = ChainSignal(
                        stage_completed=sig["stage_completed"],
                        metadata=sig.get("metadata", {}),
                    )
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

            yield AgentEvent(
                role="tool_result",
                content=tool_result[:1500],
                status=AgentPhase.TOOL_USE,
                agent_name=self.name,
                tool_name=pending["tool_name"],
                tool_result=tool_result[:1500],
                chain_signal=parsed_chain_signal,
            )

            fake_id = f"confirmed_{uuid.uuid4().hex[:8]}"
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": fake_id,
                            "name": pending["tool_name"],
                            "input": pending["tool_input"],
                        }
                    ],
                }
            )
            self.conversation.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": fake_id, "content": tool_result}],
                }
            )

            # On a chain signal, stop without a further report — the dispatcher takes over.
            if parsed_chain_signal is not None:
                self._execution_confirmed = False
                self._execution_params_seen = False
                self.status = AgentPhase.IDLE
                return

            # Report the result.
            self.conversation.append(
                {
                    "role": "user",
                    "content": "Analyse the tool result above and report it.",
                }
            )
            self._pre_run_conv_len = len(self.conversation) - 3
            self.status = AgentPhase.THINKING
            for _ in range(10):
                try:
                    response = await self._call_llm()
                except Exception as e:
                    self.status = AgentPhase.ERROR
                    yield AgentEvent(
                        role="assistant",
                        content=self._rt("agent_error", error=str(e)),
                        status=AgentPhase.ERROR,
                        agent_name=self.name,
                    )
                    return
                for block in response.content:
                    if block.type == "text":
                        self.conversation.append(
                            {"role": "assistant", "content": [{"type": "text", "text": block.text}]}
                        )
                        self.status = AgentPhase.RESPONDING
                        yield AgentEvent(
                            role="assistant", content=block.text, status=AgentPhase.RESPONDING, agent_name=self.name
                        )
                break
            self._execution_confirmed = False
            self._execution_params_seen = False
            self.status = AgentPhase.IDLE
            return

        # Conversation length before this run, used to scope tool filtering.
        self._pre_run_conv_len = len(self.conversation)
        # Guided: only allow execution when the previous assistant asked for confirmation and the user gave it.
        _is_user_confirming = False
        if self.autonomy == "guided" and any(w in _msg_lower for w in _confirm_words):
            # Was the previous assistant message a confirmation request?
            for msg in reversed(self.conversation[-6:]):
                if msg.get("role") == "assistant":
                    _ac = msg.get("content", "")
                    if isinstance(_ac, list):
                        _ac = " ".join(
                            b.get("text", "") for b in _ac if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if is_confirmation_question(_ac):
                        _is_user_confirming = True
                    break
        self._execution_confirmed = _is_user_confirming
        self._execution_params_seen = False
        self.conversation.append({"role": "user", "content": user_message})
        self.status = AgentPhase.THINKING

        yield AgentEvent(
            role="system",
            content=self._rt("analyzing"),
            status=AgentPhase.THINKING,
            agent_name=self.name,
        )

        max_iterations = 10  # tool-use loop max iterations
        _recent_tools: List[str] = []  # for repeat detection
        _MAX_SAME_TOOL = 2  # block after this many consecutive calls to one tool
        _empty_response_count = 0  # consecutive empty responses
        _MAX_EMPTY_RETRIES = 2  # retries allowed on an empty response
        _auto_advance_count = 0  # consecutive _next_step auto-runs
        _MAX_AUTO_ADVANCE = 3  # cap on auto-runs, to break alternating loops
        for _iter_i in range(max_iterations):
            self._trim_conversation()
            try:
                response = await self._call_llm()
            except Exception as e:
                self.status = AgentPhase.ERROR
                yield AgentEvent(
                    role="assistant",
                    content=self._rt("agent_error", error=str(e)),
                    status=AgentPhase.ERROR,
                    agent_name=self.name,
                )
                return

            # Process response content blocks
            assistant_content = []
            has_tool_use = False
            final_text = ""

            for block in response.content:
                if block.type == "thinking":
                    self.status = AgentPhase.THINKING
                    yield AgentEvent(
                        role="assistant",
                        content="",
                        thinking=block.thinking,
                        status=AgentPhase.THINKING,
                        agent_name=self.name,
                    )
                    assistant_content.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                        }
                    )

                elif block.type == "text":
                    final_text += block.text
                    assistant_content.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )

                elif block.type == "tool_use":
                    # ── Repeat detection ──
                    _recent_tools.append(block.name)
                    # (1) The same tool called repeatedly.
                    _same_tool_loop = len(_recent_tools) >= _MAX_SAME_TOOL and all(
                        t == block.name for t in _recent_tools[-_MAX_SAME_TOOL:]
                    )
                    # (2) Alternating repeats (A, B, A, B).
                    _cycle_detected = False
                    if len(_recent_tools) >= 4:
                        _last4 = _recent_tools[-4:]
                        # Two tools alternating: [A, B, A, B]
                        if _last4[0] == _last4[2] and _last4[1] == _last4[3] and _last4[0] != _last4[1]:
                            _cycle_detected = True
                    if _same_tool_loop or _cycle_detected:
                        import logging as _log

                        _pattern = (
                            f"cycle {_recent_tools[-4:]}" if _cycle_detected else f"'{block.name}' x{_MAX_SAME_TOOL}"
                        )
                        _log.getLogger("agents.foundation").warning(f"Tool loop detected ({_pattern}) — breaking loop")
                        # Break the loop: skip the tool and emit a message instead.
                        self.status = AgentPhase.RESPONDING
                        _loop_msg = self._rt("tool_loop_stopped")
                        _clean = [
                            b
                            for b in assistant_content
                            if not (b.get("type") == "text" and not b.get("text", "").strip())
                        ]
                        if not _clean:
                            _clean = [{"type": "text", "text": _loop_msg}]
                        self.conversation.append(
                            {
                                "role": "assistant",
                                "content": _clean,
                            }
                        )
                        yield AgentEvent(
                            role="assistant",
                            content=_loop_msg,
                            status=AgentPhase.RESPONDING,
                            agent_name=self.name,
                        )
                        has_tool_use = False  # leave the loop
                        break

                    # ── Guided: guard calls to execution tools ──
                    # Guided mode blocks an execution tool until the user confirms.
                    _GUARDED_TOOLS = EXECUTION_TOOLS
                    if (
                        self.autonomy == "guided"
                        and block.name in _GUARDED_TOOLS
                        and _should_block(
                            block.name,
                            confirmed=getattr(self, "_execution_confirmed", False),
                            params_seen=getattr(self, "_execution_params_seen", False),
                        )
                    ):
                        _tool_label = (
                            self._rt(TOOL_LABEL_KEYS[block.name]) if block.name in TOOL_LABEL_KEYS else block.name
                        )

                        # Guided: run after confirmation.
                        logger.info(
                            f"Guided mode: blocking execution tool '{block.name}' -- waiting for user confirmation"
                        )
                        # The model's text (the recommendation) becomes the final reply.
                        if final_text.strip():
                            self.conversation.append(
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": final_text}],
                                }
                            )
                            self.status = AgentPhase.RESPONDING
                            yield AgentEvent(
                                role="assistant",
                                content=final_text,
                                status=AgentPhase.RESPONDING,
                                agent_name=self.name,
                            )
                        # Store the pending execution for the next confirmation.
                        self._pending_execution = {
                            "tool_name": block.name,
                            "tool_input": block.input,
                        }
                        # Without text, emit a default message.
                        if not final_text.strip():
                            _ask_text = self._rt("tool_confirm_ask", tool_label=_tool_label)
                            self.conversation.append(
                                {"role": "assistant", "content": [{"type": "text", "text": _ask_text}]}
                            )
                            self.status = AgentPhase.RESPONDING
                            yield AgentEvent(
                                role="assistant", content=_ask_text, status=AgentPhase.RESPONDING, agent_name=self.name
                            )
                        self.status = AgentPhase.IDLE
                        return  # return, so the outer loop exits too

                    has_tool_use = True
                    _empty_response_count = 0  # a successful tool call resets the counter
                    _auto_advance_count = 0  # reset the auto-advance counter on a direct tool call
                    self.status = AgentPhase.TOOL_USE

                    yield AgentEvent(
                        role="assistant",
                        content=self._rt("tool_call", tool=block.name),
                        status=AgentPhase.TOOL_USE,
                        agent_name=self.name,
                        tool_name=block.name,
                        tool_input=block.input,
                    )

                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

                    # Execute tool
                    logger.debug(f"[TOOL-EXEC] {self.name}: executing '{block.name}' (iter={_iter_i})")
                    try:
                        tool_result = await self.execute_tool(block.name, block.input)
                        logger.debug(
                            f"[TOOL-EXEC] {self.name}: '{block.name}' completed (result_len={len(tool_result) if tool_result else 0})"
                        )
                    except Exception as e:
                        logger.error(f"[TOOL-EXEC] {self.name}: '{block.name}' EXCEPTION: {e}")
                        tool_result = f"Tool error: {str(e)}"

                    # Parse structured chain signal from tool result (if present)
                    parsed_chain_signal = None
                    try:
                        parsed = json.loads(tool_result)
                        if isinstance(parsed, dict) and "__chain_signal__" in parsed:
                            sig = parsed["__chain_signal__"]
                            parsed_chain_signal = ChainSignal(
                                stage_completed=sig["stage_completed"],
                                metadata=sig.get("metadata", {}),
                            )
                            logger.debug(
                                f"[CHAIN-PARSE] {self.name}: parsed chain_signal from '{block.name}' -> stage={parsed_chain_signal.stage_completed}"
                            )
                        else:
                            logger.debug(
                                f"[CHAIN-PARSE] {self.name}: tool '{block.name}' result has no __chain_signal__ (keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}, result_len={len(tool_result)})"
                            )
                    except (json.JSONDecodeError, TypeError, KeyError) as _e:
                        logger.debug(
                            f"[CHAIN-PARSE] {self.name}: failed to parse chain_signal from '{block.name}': {type(_e).__name__}: {_e} (result_len={len(tool_result) if tool_result else 0})"
                        )

                    yield AgentEvent(
                        role="tool_result",
                        content=tool_result[:1500],  # preview
                        status=AgentPhase.TOOL_USE,
                        agent_name=self.name,
                        tool_name=block.name,
                        tool_result=tool_result[:1500],
                        chain_signal=parsed_chain_signal,
                    )

                    # Add assistant message and tool result to conversation
                    # drop empty text blocks
                    _clean_content = [
                        b for b in assistant_content if not (b.get("type") == "text" and not b.get("text", "").strip())
                    ]
                    self.conversation.append(
                        {
                            "role": "assistant",
                            "content": _clean_content or assistant_content,
                        }
                    )
                    self.conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": tool_result,
                                }
                            ],
                        }
                    )
                    # Reset for next iteration
                    assistant_content = []

                    # A chain signal ends the agentic loop immediately;
                    # the dispatcher handles the next stage.
                    if parsed_chain_signal is not None:
                        logger.debug(
                            f"[CHAIN] {self.name}: chain_signal detected from '{block.name}' -> stage={parsed_chain_signal.stage_completed}, returning immediately"
                        )
                        self.status = AgentPhase.IDLE
                        return

            # Fallback: detect XML-style tool invocations in text response
            # Some models emit <invoke name="tool"> text instead of tool_use blocks
            if not has_tool_use and final_text and "<invoke" in final_text:
                import re as _re

                invoke_match = _re.search(
                    r'<invoke\s+name=["\']([^"\']+)["\']\s*>(.*?)</invoke>',
                    final_text,
                    _re.DOTALL,
                )
                if invoke_match:
                    tool_name_parsed = invoke_match.group(1)
                    # ── Repeat detection for the XML fallback ──
                    _recent_tools.append(tool_name_parsed)
                    if len(_recent_tools) >= _MAX_SAME_TOOL and all(
                        t == tool_name_parsed for t in _recent_tools[-_MAX_SAME_TOOL:]
                    ):
                        import logging as _log

                        _log.getLogger("agents.foundation").warning(
                            f"XML fallback tool '{tool_name_parsed}' called {_MAX_SAME_TOOL} times — breaking loop"
                        )
                        clean_text = final_text[: invoke_match.start()].strip()
                        self.status = AgentPhase.RESPONDING
                        yield AgentEvent(
                            role="assistant",
                            content=clean_text or f"Stopping: '{tool_name_parsed}' was called repeatedly.",
                            status=AgentPhase.RESPONDING,
                            agent_name=self.name,
                        )
                        break

                    invoke_body = invoke_match.group(2).strip()
                    # Parse parameters from <parameter name="key">value</parameter>
                    tool_input_parsed = {}
                    for pm in _re.finditer(
                        r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>',
                        invoke_body,
                        _re.DOTALL,
                    ):
                        tool_input_parsed[pm.group(1)] = pm.group(2).strip()
                    # If no <parameter> tags, try JSON body
                    if not tool_input_parsed and invoke_body:
                        try:
                            tool_input_parsed = json.loads(invoke_body)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Strip the XML from displayed text
                    clean_text = final_text[: invoke_match.start()].strip()

                    # ── The guided guard applies to the XML fallback too ──
                    _GUARDED_TOOLS_XML = EXECUTION_TOOLS

                    has_tool_use = True
                    self.status = AgentPhase.TOOL_USE
                    fake_id = f"xml_fallback_{uuid.uuid4().hex[:8]}"

                    yield AgentEvent(
                        role="assistant",
                        content=self._rt("tool_call", tool=tool_name_parsed),
                        status=AgentPhase.TOOL_USE,
                        agent_name=self.name,
                        tool_name=tool_name_parsed,
                        tool_input=tool_input_parsed,
                    )

                    assistant_content_fb = []
                    if clean_text:
                        assistant_content_fb.append({"type": "text", "text": clean_text})
                    assistant_content_fb.append(
                        {
                            "type": "tool_use",
                            "id": fake_id,
                            "name": tool_name_parsed,
                            "input": tool_input_parsed,
                        }
                    )

                    try:
                        tool_result = await self.execute_tool(tool_name_parsed, tool_input_parsed)
                    except Exception as e:
                        tool_result = f"Tool error: {str(e)}"

                    parsed_chain_signal = None
                    try:
                        parsed = json.loads(tool_result)
                        if isinstance(parsed, dict) and "__chain_signal__" in parsed:
                            sig = parsed["__chain_signal__"]
                            parsed_chain_signal = ChainSignal(
                                stage_completed=sig["stage_completed"],
                                metadata=sig.get("metadata", {}),
                            )
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass

                    yield AgentEvent(
                        role="tool_result",
                        content=tool_result[:1500],
                        status=AgentPhase.TOOL_USE,
                        agent_name=self.name,
                        tool_name=tool_name_parsed,
                        tool_result=tool_result[:1500],
                        chain_signal=parsed_chain_signal,
                    )

                    self.conversation.append({"role": "assistant", "content": assistant_content_fb})
                    self.conversation.append(
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": fake_id, "content": tool_result}],
                        }
                    )
                    # A chain signal ends the agentic loop; the dispatcher continues.
                    if parsed_chain_signal is not None:
                        self.status = AgentPhase.IDLE
                        return
                    # Continue the agentic loop — don't break
                    continue

            if not has_tool_use:
                    # ── Automatic tool call driven by _next_step ──
                _text_stripped = final_text.strip() if final_text else ""
                logger.debug(
                    f"[AUTO-ADVANCE-CHECK] {self.name}: no tool_use in response, text_len={len(_text_stripped)}, checking _next_step..."
                )
                if self.conversation:
                    next_tool_name = self._extract_next_step_hint()
                    logger.debug(f"[AUTO-ADVANCE-CHECK] {self.name}: _next_step hint -> {next_tool_name}")
                    if next_tool_name and _auto_advance_count >= _MAX_AUTO_ADVANCE:
                        import logging as _log_aa

                        _log_aa.getLogger("agents.foundation").warning(
                            f"Auto-advance limit reached ({_MAX_AUTO_ADVANCE}) — stopping auto tool chain to prevent loop"
                        )
                        logger.debug(
                            f"[AUTO-ADVANCE] {self.name}: limit reached ({_MAX_AUTO_ADVANCE} consecutive), stopping"
                        )
                        next_tool_name = None  # stop auto-running
                    if next_tool_name:
                        # auto: execution tools run automatically, no confirmation needed
                        # guided/manual: execution tools are excluded from auto-advance
                        _skip_execution = next_tool_name in EXECUTION_TOOLS and self.autonomy != "auto"
                        if (
                            next_tool_name
                            and next_tool_name in [t.get("name") for t in self.tools]
                            and not _skip_execution
                        ):
                            logger.debug(
                                f"[AUTO-ADVANCE] {self.name}: Force-executing '{next_tool_name}' via _next_step (params={getattr(self, '_next_step_params', {})})"
                            )
                            # ── Run the tool from code, independent of the model ──
                            _auto_advance_count += 1
                            fake_id = f"auto_{uuid.uuid4().hex[:8]}"
                            # Use the parameters extracted from _next_step.
                            auto_params = getattr(self, "_next_step_params", {})
                            # Record the model's text in the conversation.
                            auto_assistant = []
                            if _text_stripped:
                                auto_assistant.append({"type": "text", "text": final_text})
                            auto_assistant.append(
                                {
                                    "type": "tool_use",
                                    "id": fake_id,
                                    "name": next_tool_name,
                                    "input": auto_params,
                                }
                            )

                            yield AgentEvent(
                                role="assistant",
                                content=self._rt("tool_call", tool=next_tool_name),
                                status=AgentPhase.TOOL_USE,
                                agent_name=self.name,
                                tool_name=next_tool_name,
                                tool_input=auto_params,
                            )

                            try:
                                tool_result = await self.execute_tool(next_tool_name, auto_params)
                            except Exception as e:
                                tool_result = f"Tool error: {str(e)}"

                            # Parse the chain signal.
                            parsed_chain_signal = None
                            try:
                                parsed = json.loads(tool_result)
                                if isinstance(parsed, dict) and "__chain_signal__" in parsed:
                                    sig = parsed["__chain_signal__"]
                                    parsed_chain_signal = ChainSignal(
                                        stage_completed=sig["stage_completed"],
                                        metadata=sig.get("metadata", {}),
                                    )
                            except (json.JSONDecodeError, TypeError, KeyError):
                                pass

                            yield AgentEvent(
                                role="tool_result",
                                content=tool_result[:1500],
                                status=AgentPhase.TOOL_USE,
                                agent_name=self.name,
                                tool_name=next_tool_name,
                                tool_result=tool_result[:1500],
                                chain_signal=parsed_chain_signal,
                            )

                            self.conversation.append({"role": "assistant", "content": auto_assistant})
                            self.conversation.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "tool_result", "tool_use_id": fake_id, "content": tool_result}
                                    ],
                                }
                            )
                            _recent_tools.append(next_tool_name)
                            assistant_content = []
                            has_tool_use = True
                            # A chain signal ends the agentic loop; the dispatcher continues.
                            if parsed_chain_signal is not None:
                                self.status = AgentPhase.IDLE
                                return
                            continue

                    # Guard against an empty response after a tool result.
                    if not _text_stripped or _text_stripped == "(no response)":
                        last_msg = self.conversation[-1]
                        last_content = last_msg.get("content", "")
                        has_pending_tool = False
                        if isinstance(last_content, list):
                            has_pending_tool = any(
                                isinstance(b, dict) and b.get("type") == "tool_result" for b in last_content
                            )
                        if has_pending_tool:
                            _empty_response_count += 1
                            if _empty_response_count <= _MAX_EMPTY_RETRIES:
                                continue
                            # Out of retries: force the _next_step hint if there is one, else stop.
                            _forced_tool = (
                                self._extract_next_step_hint() if hasattr(self, "_extract_next_step_hint") else None
                            )
                            if (
                                _forced_tool
                                and _forced_tool in [t["name"] for t in self.tools]
                                and _auto_advance_count < _MAX_AUTO_ADVANCE
                            ):
                                logger.debug(
                                    f"[{self.name}] Empty response limit reached, force-calling: {_forced_tool}"
                                )
                                try:
                                    _auto_advance_count += 1
                                    _forced_input = (
                                        self._build_next_step_input(_forced_tool)
                                        if hasattr(self, "_build_next_step_input")
                                        else {}
                                    )
                                    _forced_result = await self.execute_tool(_forced_tool, _forced_input)
                                    yield AgentEvent(
                                        role="tool_result",
                                        content=_forced_result or "",
                                        status=AgentPhase.RESPONDING,
                                        agent_name=self.name,
                                        tool_name=_forced_tool,
                                    )
                                    self.conversation.append(
                                        {
                                            "role": "assistant",
                                            "content": [
                                                {
                                                    "type": "tool_use",
                                                    "id": f"forced_{_forced_tool}",
                                                    "name": _forced_tool,
                                                    "input": _forced_input,
                                                }
                                            ],
                                        }
                                    )
                                    self.conversation.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "tool_result",
                                                    "tool_use_id": f"forced_{_forced_tool}",
                                                    "content": _forced_result or "",
                                                }
                                            ],
                                        }
                                    )
                                    _empty_response_count = 0
                                    # Parse the chain signal.
                                    try:
                                        _fr_parsed = json.loads(_forced_result) if _forced_result else {}
                                        if "__chain_signal__" in _fr_parsed:
                                            _cs = _fr_parsed["__chain_signal__"]
                                            _chain_ev = AgentEvent(
                                                role="tool_result",
                                                content=_forced_result,
                                                status=AgentPhase.RESPONDING,
                                                agent_name=self.name,
                                                tool_name=_forced_tool,
                                            )
                                            _chain_ev.chain_signal = ChainSignal(
                                                stage_completed=_cs.get("stage_completed", ""),
                                                metadata=_cs.get("metadata", {}),
                                            )
                                            yield _chain_ev
                                            return
                                    except Exception:
                                        pass
                                    continue
                                except Exception as _fe:
                                    logger.error(f"[{self.name}] Force-call failed: {_fe}")
                            # No _next_step either — leave the loop.
                            logger.debug(f"[{self.name}] Empty response limit reached, no fallback -- stopping")
                            break

                # No tool use and no auto-advance — final response
                logger.debug(
                    f"[LOOP-END] {self.name}: iteration={_iter_i}, breaking with text response (len={len(final_text.strip()) if final_text else 0})"
                )
                self.conversation.append(
                    {
                        "role": "assistant",
                        "content": assistant_content if assistant_content else final_text,
                    }
                )
                self.status = AgentPhase.RESPONDING
                yield AgentEvent(
                    role="assistant",
                    content=final_text,
                    status=AgentPhase.RESPONDING,
                    agent_name=self.name,
                )
                break

        self.status = AgentPhase.IDLE

    def _extract_next_step_hint(self) -> Optional[str]:
        """Tool name from the _next_step hint in the previous tool result.

        When the tool result JSON carries a _next_step field naming the next
        tool, return just that name.
        """
        import re as _re

        for msg in reversed(self.conversation[-8:]):
            content = msg.get("content", "")
            raw_candidates = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        raw_candidates.append(block.get("content", ""))
            elif isinstance(content, str) and '"_next_step"' in content:
                raw_candidates.append(content)

            for raw in raw_candidates:
                try:
                    data = json.loads(raw)
                    hint = data.get("_next_step", "")
                    if hint:
                        match = _re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", hint)
                        if match:
                            tool_name = match.group(1)
                            # Extract parameters: "detect_document_type(raw_dataset_folder='xxx')"
                            param_match = _re.search(r"\(([^)]+)\)", hint)
                            if param_match:
                                params = {}
                                for kv in _re.finditer(
                                    r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]",
                                    param_match.group(1),
                                ):
                                    params[kv.group(1)] = kv.group(2)
                                self._next_step_params = params
                            else:
                                self._next_step_params = {}
                            return tool_name
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        # Fallback: parse the tool name straight out of a plain text response,
        # e.g. "I will call unified_assessment" -> "unified_assessment".
        return None

    def _trim_conversation(self):
        """Trim conversation history to prevent unbounded memory growth.

        If conversation exceeds 40 messages, keep only the system prompt
        (first message) and the last 20 messages.
        """
        _MAX_MESSAGES = 40
        _KEEP_RECENT = 20
        if len(self.conversation) > _MAX_MESSAGES:
            first_msg = self.conversation[0] if self.conversation else None
            recent = self.conversation[-_KEEP_RECENT:]
            if first_msg and first_msg.get("role") == "system":
                self.conversation = [first_msg] + recent
            else:
                self.conversation = recent

    async def _call_llm(self) -> Any:
        """LLM call for the tool loop."""
        _ = self.client
        return await self._call_gpt()

    async def _call_gpt(self) -> Any:
        """Call OpenAI chat completions and normalise the reply into internal response objects."""
        import json as _json

        # JSON-schema tool spec → OpenAI function format
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in self._effective_tools()
        ]

        # Build OpenAI messages from the conversation (internal format → OpenAI format)
        effective_system = self._effective_system_prompt()
        oai_messages: List[Dict[str, Any]] = [{"role": "system", "content": effective_system}]

        for msg in self.conversation:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                continue

            if isinstance(content, str):
                oai_messages.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                continue

            # tool_result blocks → OpenAI "tool" role
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for tr in tool_results:
                    oai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", "unknown"),
                            "content": tr.get("content", ""),
                        }
                    )
                continue

            # assistant message: may have text + tool_use blocks
            if role == "assistant":
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                oai_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts).strip() or None,
                }
                if tool_uses:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tu.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tu.get("name", ""),
                                "arguments": _json.dumps(tu.get("input", {}), ensure_ascii=False),
                            },
                        }
                        for i, tu in enumerate(tool_uses)
                    ]
                oai_messages.append(oai_msg)
                continue

            # user message with content list (plain text only)
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if text_parts:
                oai_messages.append({"role": role, "content": "\n".join(text_parts).strip()})

        # Call OpenAI API
        kwargs: Dict[str, Any] = {
            "model": GPT_MODEL,
            "max_tokens": self.max_tokens,
            "messages": oai_messages,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        loop = asyncio.get_event_loop()
        oai_response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(**kwargs),
        )

        # Convert the OpenAI response → internal _Response/_Block objects
        class _Block:
            def __init__(self, btype, **kw):
                self.type = btype
                for k, v in kw.items():
                    setattr(self, k, v)

        class _Response:
            def __init__(self, blocks, stop):
                self.content = blocks
                self.stop_reason = stop

        blocks = []
        choice_msg = oai_response.choices[0].message

        if choice_msg.content:
            blocks.append(_Block("text", text=choice_msg.content))

        if choice_msg.tool_calls:
            for tc in choice_msg.tool_calls:
                try:
                    input_data = _json.loads(tc.function.arguments)
                except (_json.JSONDecodeError, TypeError):
                    input_data = {}
                blocks.append(
                    _Block(
                        "tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=input_data,
                    )
                )

        stop_reason = "tool_use" if choice_msg.tool_calls else "end_turn"
        if not blocks:
            blocks.append(_Block("text", text="(no response)"))

        return _Response(blocks, stop_reason)

    async def complete_text(self, prompt: str, max_tokens: int = 8192) -> str:
        """Single-turn completion without tools. Used for judgments and summaries."""
        _ = self.client
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        return response.choices[0].message.content.strip()

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "conversation_length": len(self.conversation),
            "model": self.model,
            "autonomy": self.autonomy,
            "agent_provider": "openai",
        }
