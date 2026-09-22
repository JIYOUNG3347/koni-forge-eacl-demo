"""
KONI-Forge AgentDispatcher — Task distribution and agent coordination

Analyzes user requests and routes to appropriate agents,
automatically chains next steps upon agent completion.

Pipeline (full sequence):
  retrieval (indexing) -> boundary (KBD) -> tuning (training)

Each step is independent — users can enter at any point.
Chaining is triggered by structured __chain_signal__ in tool results,
not by fragile text matching.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from server.core import train_rec_prompt as _train_rec_prompt
from server.core.agent_confirm import confirm_hint as _confirm_hint
from server.core.agent_confirm import texts as _ct
from server.core.agent_handoff import handoff as _handoff
from server.core.agent_intent_words import is_confirmation as _intent_is_confirmation
from server.core.agent_intent_words import is_question as _intent_is_question
from server.core.agent_intent_words import is_rejection as _intent_is_rejection
from server.core.agent_intent_words import is_skip as _intent_is_skip
from server.core.agent_runtime_texts import text as _rt
from server.core.auto_intent import DECISION_MAX_TOKENS
from server.core.config import STORAGE_ROOT
from server.core.corpus_files import select_data_files
from server.core.dataset_paths import dataset_roots
from server.core.lang import CONFIG_KEY as _LANG_CONFIG_KEY
from server.core.llm_tokens import JSON_TOKENS
from server.core.orchestration import (
    append_trace,
    build_plan_prompt,
    build_planner_prompt,
    default_pipeline_graph,
    edge_is_wireable,
    execute_event,
    fallback_route,
    from_session,
    initial_route_candidates,
    is_auto_planner,
    is_graph_mode,
    parse_plan_response,
    parse_planner_decision,
    plan_event,
    plan_to_dict,
    plan_to_graph,
    resolve_route,
    route_event,
    summarize_traverse,
    transition_event,
)
from server.core.orchestration.blackboard import _FT_KEYWORDS
from server.core.pipeline_marker import AUTO_MARKER, has_pipeline_marker
from server.core.train_target import (
    STATUS_ABSENT,
    STATUS_AMBIGUOUS,
    dataset_names,
    resolve_requested_model,
    resolve_training_dataset,
)

from .foundation import CHROMA_DIR, AgentBase, AgentEvent, AgentPhase, ChainSignal

logger = logging.getLogger("agents.dispatcher")

_OUTPUTS = STORAGE_ROOT / "outputs"

# ──────────────────────────────────────────────────────────
# Pipeline stage definitions
# ──────────────────────────────────────────────────────────
# Each stage knows its next step, which agent handles it,
# and what message to send.

PIPELINE_STAGES: Dict[str, Dict[str, Any]] = {
    "retrieval": {
        "next_agent": "boundary",
        "next_message": "",  # built by _chain_texts
        "announce": "",
    },
    "kbd": {
        "resolver": "_resolve_kbd_path",
    },
    "tuning": {
        "next_agent": None,
        "next_message": "",
        "announce": "",
    },
}

# Chain used when KBD is skipped and training follows directly.
_TUNING_SKIP_CHAIN: Dict[str, Any] = {
    "next_agent": "tuning",
    "next_message_key": "skip_next",
    "announce_key": "announce_skip_to_training",
}

# Stage to land on after skipping the proposed agent. None means nothing left to skip to.
_SKIP_LAND_TARGET: Dict[str, Optional[str]] = {
    "boundary": "tuning",  # skip KBD -> propose training directly
    "tuning": None,  # skip training -> end of chain
}


def decide_skip(next_agent: Optional[str]) -> Dict[str, Any]:
    """Decide what happens when the proposed agent (pending_chain.next_agent) is skipped.

    Pure function — testable without an AgentDispatcher instance.

    Returns:
        {"action": "advance"|"end", "agent": next_agent, "land": Optional[str]}
        - advance: propose the card for the landing stage.
        - end    : nothing left to skip to (terminal).
    """
    if next_agent in _SKIP_LAND_TARGET:
        land = _SKIP_LAND_TARGET[next_agent]
        return {
            "action": "end" if land is None else "advance",
            "agent": next_agent,
            "land": land,
        }
    # None (RAG terminal and similar) or an unknown agent — end gracefully.
    return {"action": "end", "agent": next_agent, "land": None}


# Maximum chaining depth to prevent infinite loops
_MAX_CHAIN_DEPTH = 7

_DEFAULT_MAX_SEQ_LEN = 2048
_DEFAULT_LORA_R = 8
_DEFAULT_LORA_ALPHA = 16
_DEFAULT_LORA_DROPOUT = 0.1
# KBD coverage threshold (%) for "very low" — shared by the full-FT recommendation
_LOW_COVERAGE_PCT = 30

_KBD_PRESERVED_KEYS = (
    "recommendation",
    "knowledge_coverage_pct",
    "hallucination_rate_pct",
    "weak_categories",
    "boundary_categories",
    "doc_name",
    "model_name",
)


def _new_session_dict(mode: str) -> Dict[str, Any]:
    import time as _t

    return {
        "session_id": f"s-{int(_t.time())}",
        "mode": mode,
        "started_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": [],
    }


_ROUTING_TABLE: list[tuple[list[str], str]] = [
    (["kbd", "knowledge boundary", "probe", "coverage", "hallucination"], "boundary"),
    (["train", "fine-tun", "lora", "fft", "epoch", "learning rate", "checkpoint"], "tuning"),
    (["rag", "retriev", "index", "embedding", "vector", "chroma", "upload"], "retrieval"),
]

_STAGE_ORDER: tuple = ("retrieval", "boundary", "tuning")


def stage_label(agent: str, lang: Optional[str] = None) -> str:
    """Agent name to a human-readable stage name. Unknown values pass through."""
    from server.core.agent_runtime_texts import text

    label = text(f"stage_label_{agent}", lang)
    return agent if label == f"stage_label_{agent}" else label


def is_rejection(message: str, lang: Optional[str] = None) -> bool:
    """Whether the message declines. Used for declined DPO proposals."""
    return _intent_is_rejection(message, lang)


def route_match(message: str) -> Optional[str]:
    """Keyword routing — the agent name on a match, None otherwise.

    Unlike ``_route()``, which falls through to 'corpus', this returns None so a
    natural-language resume can tell 'not recognised' apart from a default.
    """
    msg = message.lower()
    for keywords, agent_name in _ROUTING_TABLE:
        if any(kw in msg for kw in keywords):
            return agent_name
    return None


def is_question(message: str, lang: Optional[str] = None) -> bool:
    """Whether the message is a question — a "?" or a question marker.

    Guards against a clarifying question ("what is KBD?") being taken as a
    resume and running the specialist.
    """
    return _intent_is_question(message, lang)


def classify_resume(
    matched_agent: Optional[str],
    pending_agent: Optional[str],
    override_agent: Optional[str],
) -> str:
    """Classify a natural-language resume command.

    - "no_pending" : no stage is waiting; dispatch normally.
    - "unmatched"  : matches no stage keyword; explain and keep the card.
    - "matched"    : matches the proposed stage; resume.
    - "override"   : the user repeats the stage we just asked about; proceed.
    - "mismatched" : a different stage from the proposal; ask about the order.
    """
    if pending_agent is None:
        return "no_pending"
    if matched_agent is None:
        return "unmatched"
    if matched_agent == pending_agent:
        return "matched"
    if override_agent is not None and matched_agent == override_agent:
        return "override"
    return "mismatched"


# (Legacy path — replaced by the per-user _session_path.)


class AgentDispatcher:
    """
    Agent dispatcher, one instance per user.

    Roles:
    - Analyzes user requests and routes to appropriate agents
    - Manages agent instances (per-user isolated)
    - Tracks pipeline state (current stage, history, pending chains)
    - Automatically chains next steps upon agent completion (auto mode)
    - Stores pending chain for user confirmation (guided mode)
    - Persists pipeline session to disk for assessment page
    """

    def __init__(self, user_id: str = "default"):
        self._user_id = user_id
        # Per-user storage path.
        self._storage_root = _OUTPUTS / user_id
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._session_path = self._storage_root / "pipeline_session.json"

        self._pipeline_run_id: Optional[str] = None
        self._pipeline_source_folder: Optional[str] = None
        self._pipeline_train_job: Optional[str] = None
        self._agents: Dict[str, AgentBase] = {}
        self._initialize_agents()
        # Pipeline state
        self._pending_chain: Optional[Dict[str, Any]] = None
        self._current_stage: Optional[str] = None
        self._chain_history: List[str] = []
        self._stage_summaries: Dict[str, Dict[str, Any]] = {}
        self._last_agent: Optional[str] = None
        self._chain_paused: bool = False
        self._resume_override_pending: Optional[str] = None
        self._kbd_metadata: Optional[Dict[str, Any]] = None
        self._last_user_message: str = ""
        # Transition graph — used instead of PIPELINE_STAGES in graph mode.
        self._pipeline_graph = default_pipeline_graph()
        self._session: Dict[str, Any] = self._load_or_create_session()
        self._is_busy: bool = False
        self._busy_agent: Optional[str] = None
        self._busy_since: float = 0  # time.time() when busy started

    def _initialize_agents(self):
        """Create the specialist instances with user-scoped storage."""
        from .specialists.boundary_specialist import BoundarySpecialist
        from .specialists.retrieval_specialist import RetrievalSpecialist
        from .specialists.tuning_specialist import TuningSpecialist

        for agent_cls in (RetrievalSpecialist, BoundarySpecialist, TuningSpecialist):
            agent = agent_cls()
            agent.set_storage_root(self._storage_root)
            agent._user_id = self._user_id
            agent._pipeline_run_id = self._pipeline_run_id
            agent._pipeline_source_folder = self._pipeline_source_folder
            agent._pipeline_train_job = self._pipeline_train_job
            self._agents[agent.name] = agent

    def set_pipeline_run_id(self, job_id: Optional[str]) -> None:
        """Propagate the auto-pipeline run_id to the dispatcher and every agent."""
        self._pipeline_run_id = job_id
        for agent in self._agents.values():
            agent._pipeline_run_id = job_id

    def set_pipeline_source_folder(self, folder_name: Optional[str]) -> None:
        """Assigning the field directly does not reach the agents — always use
        this setter (same reason as ``set_pipeline_run_id``).

        """
        folder = (folder_name or "").strip() or None
        self._pipeline_source_folder = folder
        for agent in self._agents.values():
            agent._pipeline_source_folder = folder

    def set_pipeline_train_job(self, job_name: Optional[str]) -> None:
        """Assigning the field directly does not reach the agents — always use
        this setter (same reason as ``set_pipeline_run_id``).

        """
        job = (job_name or "").strip() or None
        self._pipeline_train_job = job
        for agent in self._agents.values():
            agent._pipeline_train_job = job

    def set_train_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        """Propagate pipeline-requested training parameters to every agent."""
        self._train_overrides = dict(overrides or {})
        for agent in self._agents.values():
            agent._pipeline_train_overrides = self._train_overrides

    def set_train_identity(self, dataset: str = "", base_model: str = "") -> None:
        """Fill in a training argument the LLM left empty.

        Unlike ``set_train_overrides``, which overwrites hyperparameters, a value
        the LLM passed properly is never pushed aside by a pipeline estimate. An
        empty value does not overwrite either: each stage knows different things,
        so the boundary stage can set the model without clearing the dataset.
        """
        current = dict(getattr(self, "_train_identity", None) or {})
        if dataset:
            current["dataset"] = dataset
        if base_model:
            current["base_model"] = base_model
        self._train_identity = current
        for agent in self._agents.values():
            agent._pipeline_train_identity = current

    @property
    def agents(self) -> Dict[str, AgentBase]:
        return self._agents

    def get_agent(self, name: str) -> Optional[AgentBase]:
        return self._agents.get(name)

    def list_agents(self):
        return [agent.get_status() for agent in self._agents.values()]

    # ──────────────────────────────────────────────────────
    # Session persistence — the evaluation page reads the pipeline path from it.
    # ──────────────────────────────────────────────────────
    def _load_session_mode(self) -> str:
        """Session mode, in order: user_config, system_config, then "guided".

        agent_autonomy is user-scoped, so the per-user user_config.json wins and
        a global system_config value cannot override a personal setting.
        """
        try:
            user_cfg_path = self._storage_root / "user_config.json"
            if user_cfg_path.exists():
                user_data = json.loads(user_cfg_path.read_text(encoding="utf-8"))
                mode = user_data.get("agent_autonomy")
                if mode:
                    return mode
        except Exception:
            pass
        try:
            from modules.agents.foundation import AgentBase

            sys_cfg = AgentBase._load_system_config()
            mode = sys_cfg.get("agent_autonomy")
            if mode:
                return mode
        except Exception:
            pass
        return "guided"

    def _load_or_create_session(self) -> Dict[str, Any]:
        """Always start a fresh session, so the UI can detect a restart and reset the chat."""
        session = _new_session_dict(self._load_session_mode())
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        return session

    def _persist_session(self):
        """Persist the session to disk (atomic write via tmp + rename)."""
        try:
            self._session_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._session_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(self._session, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self._session_path)
        except Exception:
            pass

    def _record_stage(self, stage: str, metadata: dict, skipped: bool = False, skip_reason: str = ""):
        """Record a stage in the session."""
        import time as _t

        entry: Dict[str, Any] = {
            "stage": stage,
            "skipped": skipped,
        }
        if skipped:
            entry["skip_reason"] = skip_reason
            entry["skipped_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            entry["completed_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
            # Copy the main metadata.
            if stage == "retrieval":
                entry["indexed_count"] = metadata.get("indexed_count", 0)
                entry["doc_name"] = metadata.get("doc_name", "")
            elif stage == "kbd":
                entry["recommendation"] = metadata.get("recommendation", "")
                entry["coverage_pct"] = metadata.get("knowledge_coverage_pct", 0)
                entry["hallucination_rate_pct"] = metadata.get("hallucination_rate_pct", 0)
                entry["model_name"] = metadata.get("model_name", "")
            elif stage == "tuning":
                entry["model_name"] = metadata.get("model_name", "")
                entry["method"] = metadata.get("method", "")
                entry["training_job_id"] = metadata.get("task_id", "")
        self._session["stages"] = [s for s in self._session.get("stages", []) if s["stage"] != stage]
        self._session["stages"].append(entry)
        # Refresh session["mode"] — user_config wins over the global value,
        # which is why this goes through _load_session_mode().
        self._session["mode"] = self._load_session_mode()
        self._persist_session()

    # ──────────────────────────────────────────────────────
    # Pause / Resume chaining
    # ──────────────────────────────────────────────────────
    def pause_chain(self):
        """Pause chaining. The pending chain is kept."""
        self._chain_paused = True

    def resume_chain(self):
        """Resume chaining."""
        self._chain_paused = False

    # ──────────────────────────────────────────────────────
    # Apply user-edited parameter overrides to the pending chain
    # ──────────────────────────────────────────────────────
    def apply_chain_overrides(self, overrides: Dict[str, Any]) -> None:
        """Merge user-edited params into the pending chain and rebuild the
        baked instruction (`next_message`) so the specialist uses the edited
        values verbatim.

        Silently no-ops when there's no pending chain or no editable stage.
        """
        if not self._pending_chain or not isinstance(overrides, dict) or not overrides:
            return

        chain = self._pending_chain
        params = dict(chain.get("recommended_params") or {})
        stage = params.get("stage")

        # Merge overrides into recommended_params (UI-facing snapshot)
        for k, v in overrides.items():
            params[k] = v
        chain["recommended_params"] = params

        # Rebuild the baked next_message per stage so the specialist honours
        # the edits (the LLM reads these values directly from the message).
        if stage == "tuning":
            method = str(params.get("method") or "lora").lower()
            base_model = params.get("base_model") or ""
            dataset = params.get("dataset") or params.get("dataset_name") or ""
            job_name = params.get("job_name") or (f"{dataset}_training" if dataset else "")
            epochs = params.get("epochs") or 3
            batch_size = params.get("batch_size") or 4
            learning_rate = params.get("learning_rate") or "2e-5"
            max_seq_length = params.get("max_seq_length") or _DEFAULT_MAX_SEQ_LEN

            lora_bits = ""
            if method == "lora":
                lora_r = params.get("lora_r") or _DEFAULT_LORA_R
                lora_alpha = params.get("lora_alpha") or _DEFAULT_LORA_ALPHA
                lora_dropout = (
                    params.get("lora_dropout") if params.get("lora_dropout") is not None else _DEFAULT_LORA_DROPOUT
                )
                lora_bits = f", lora_r={lora_r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}"

            chain["next_message"] = (
                f"{_ct(self._exec_lang())['confirmed_below']}\n"
                f"{_ct(self._exec_lang())['exact_params_label']}base_model='{base_model}', "
                f"dataset='{dataset}', "
                f"job_name='{job_name}', "
                f"method='{method}', "
                f"epochs={epochs}, "
                f"batch_size={batch_size}, "
                f"learning_rate='{learning_rate}', "
                f"max_seq_length={max_seq_length}"
                f"{lora_bits}"
            )

    def _pending_execution_payload(self) -> Optional[Dict[str, Any]]:
        """Evaluation tools have their own selection modal, so they are excluded:
        two modals at once leaves the user unsure which to answer.

        """
        from server.core.pending_execution import selection_payload

        for name, agent in (self._agents or {}).items():
            payload = selection_payload(name, getattr(agent, "_pending_execution", None))
            if payload:
                return payload
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Returns the current pipeline state for frontend display."""
        import os as _os
        import time as _time

        # Automatic release of a stuck busy flag:
        # 1) assessment finished (per chain_history or the session stages)
        # 2) TRAIN_TIMEOUT_SECONDS elapsed
        #    (the old 180s was shorter than the 3600s probe limit, so the KBD icon went dark)
        if self._is_busy:
            assessment_done = "tuning" in self._chain_history
            if assessment_done:
                self._is_busy = False
                self._busy_agent = None
                self._busy_since = 0
            elif self._busy_since > 0 and (_time.time() - self._busy_since) > int(
                _os.getenv("TRAIN_TIMEOUT_SECONDS", "7200")
            ):
                logger.debug(
                    f"[DISPATCHER] Auto-clearing stuck _is_busy (elapsed {_time.time() - self._busy_since:.0f}s)"
                )
                self._is_busy = False
                self._busy_agent = None
                self._busy_since = 0

        return {
            "current_stage": self._current_stage,
            "completed_stages": list(self._chain_history),
            "pending_chain": self._pending_chain["announce"] if self._pending_chain else None,
            "pending_chain_agent": self._pending_chain.get("next_agent") if self._pending_chain else None,
            "pending_chain_params": (self._pending_chain.get("recommended_params") if self._pending_chain else None),
            "pending_execution": self._pending_execution_payload(),
            "stage_summaries": dict(self._stage_summaries),
            "chain_paused": self._chain_paused,
            "is_busy": self._is_busy,
            "busy_agent": self._busy_agent,
            "session": {
                "session_id": self._session.get("session_id"),
                "mode": self._session.get("mode"),
                "stages": self._session.get("stages", []),
            },
        }

    def _list_eval_model_candidates(self, proposed: Optional[str]) -> List[Dict[str, str]]:
        """Collect candidate benchmark models, best effort.

        Even if model resolution fails, the model the agent proposed stays a
        candidate so the modal always works.
        """
        out: List[Dict[str, str]] = []
        seen: set = set()

        def _add(mid: str, kind: str) -> None:
            mid = (mid or "").strip()
            if not mid or mid in seen:
                return
            seen.add(mid)
            out.append({"id": mid, "label": mid, "kind": kind})

        if proposed:
            _add(proposed, "proposed")
        try:
            from pathlib import Path as _P

            user_id = self._storage_root.name
            from server.core.config import CHECKPOINTS_DIR as _CKPT  # local import

            # Trained checkpoints.
            _job_root = _P(_CKPT) / user_id
            if _job_root.is_dir():
                for d in sorted(_job_root.iterdir()):
                    if d.is_dir():
                        _add(d.name, "trained")
            # Base models.
            _models = STORAGE_ROOT / "models"
            if _models.is_dir():
                for d in sorted(_models.iterdir()):
                    if d.is_dir():
                        _add(d.name, "base")
        except Exception as e:  # noqa: BLE001 — failing to collect candidates is not fatal
            logger.debug(f"[DISPATCHER] eval model candidate scan failed: {e}")
        return out[:30]

    # Short exchanges answered directly, without routing to a specialist.
    _CHITCHAT_PATTERNS: list[tuple[list[str], str]] = [
        (["hello", "hi ", "hi!", "hey", "good morning"], "Hello! What can I help you with?"),
        (["thanks", "thank you", "appreciate"], "Glad it helped. Just say the word if you need anything else."),
        (["bye", "goodbye", "see you"], "Goodbye!"),
        (["nice work", "well done", "great", "perfect", "awesome"], "Thank you! Let me know what else you need."),
        (["ok", "okay", "got it", "cool"], "Sure — what would you like to do?"),
    ]
    # A message containing any of these is real work, never small talk.
    _CHITCHAT_DOMAIN_KEYWORDS: list[str] = [
        "dataset",
        "data",
        "model",
        "train",
        "fine",
        "document",
        "pdf",
        "docx",
        "corpus",
        "gpu",
        "eval",
        "rag",
        "pipeline",
        "generate",
        "analy",
        "search",
        "retriev",
        "embedding",
    ]

    def _is_chitchat(self, message: str) -> bool:
        """Whether this is small talk: no domain keyword and a short chitchat pattern."""
        msg = message.strip().lower()
        if not msg:
            return False
        if any(kw in msg for kw in self._CHITCHAT_DOMAIN_KEYWORDS):
            return False
        if len(msg) <= 25:
            for patterns, _ in self._CHITCHAT_PATTERNS:
                if any(p in msg for p in patterns):
                    return True
        return False

    def _chitchat_reply(self, message: str) -> str:
        """Direct reply to small talk."""
        msg = message.strip().lower()
        for patterns, reply in self._CHITCHAT_PATTERNS:
            if any(p in msg for p in patterns):
                return reply
        return "Sure — what would you like to do?"

    # ──────────────────────────────────────────────────────
    # Routing
    # ──────────────────────────────────────────────────────

    def _route(self, message: str) -> str:
        """Determines appropriate agent based on message content (keyword matching)."""
        return fallback_route(route_match(message), self._last_agent, self._agents.keys())

    def _last_reply_tail(self, limit: int = 200) -> Optional[str]:
        """Tail of the previous agent's last assistant reply, as routing context.

        Lets a short answer to a question like "shall I run it?" be read in context.
        Failure is harmless (returns None, and the prompt omits the section).
        """
        agent = self._agents.get(self._last_agent) if self._last_agent else None
        conversation = getattr(agent, "conversation", None)
        if not conversation:
            return None
        for turn in reversed(conversation):
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            content = turn.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(t for t in texts if t)
            if isinstance(content, str) and content.strip():
                return content.strip()[-limit:]
        return None

    def _planner_agent(self):
        """An agent to call the supervisor planner with (any has complete_text).
        Prefers assessment, otherwise the first available."""
        return self._agents.get("boundary") or next(iter(self._agents.values()), None)

    def _list_raw_folders(self, limit: int = 40) -> List[str]:
        """Raw dataset folder names, newest first. Empty on failure (nothing starts)."""
        try:
            raw_dir = STORAGE_ROOT / "raw_corpus"
            if not raw_dir.is_dir():
                return []
            folders = sorted(
                [d for d in raw_dir.iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return [d.name for d in folders[:limit]]
        except Exception:  # noqa: BLE001 — a lookup failure must not kill the conversation
            return []

    async def _route_smart(self, message: str) -> str:
        """Initial routing — the LLM supervisor in graph mode, legacy keywords otherwise.

        The supervisor is only called when ORCHESTRATOR_MODE=graph. On failure, an
        ambiguous answer, or one outside the candidates, it falls back to legacy
        LLM calls go through complete_text.
        """
        if not is_graph_mode():
            result = self._route(message)
            self._trace(route_event(mode="legacy", decision=result, legacy_decision=result, message_preview=message))
            return result

        candidates = initial_route_candidates()
        planner = self._planner_agent()
        decision = None
        if planner is not None:
            try:
                prompt = build_planner_prompt(
                    from_session(self._session),
                    message,
                    candidates,
                    last_agent=self._last_agent,
                    last_reply_tail=self._last_reply_tail(),
                    lang=self._exec_lang(),
                )
                raw = await planner.complete_text(prompt, max_tokens=DECISION_MAX_TOKENS)
                if not (raw or "").strip():
                    logger.warning(
                        "[Supervisor] empty response (likely out of reasoning tokens) — falling back to legacy"
                    )
                decision = parse_planner_decision(raw, candidates)
            except Exception as exc:  # noqa: BLE001 — routing must never die
                logger.warning(f"[Supervisor] routing failed, falling back to legacy: {exc}")

        legacy = self._route(message)
        result = resolve_route(
            graph_mode=True,
            planner_decision=decision,
            known_agents=self._agents.keys(),
            legacy=legacy,
        )
        self._trace(
            route_event(
                mode="graph",
                decision=result,
                planner_decision=decision,
                legacy_decision=legacy,
                message_preview=message,
            )
        )
        if decision and result == decision:
            logger.info(f"[Supervisor] graph routing: {message[:40]!r} -> {result}")
        return result

    def _trace(self, event: Dict[str, Any]) -> None:
        """Record a traverse event in the session. A failed record never affects execution."""
        try:
            append_trace(self._session, event)
        except Exception:  # noqa: BLE001 — recording must never kill execution
            pass

    async def _maybe_plan_workflow(self, goal: str, prev_traverse: Optional[List[dict]] = None) -> None:
        """Build the workflow plan when an auto pipeline starts.

        Only with PLANNER_MODE=auto and ORCHESTRATOR_MODE=graph does the supervisor
        generate a plan and replace self._pipeline_graph. The graph is reset to the
        static one before each attempt, so a previous plan cannot leak into a new run.
        A failure at any step leaves the static graph in place, and the result is
        recorded as a traverse plan event and the session "plan" key.

        The previous run's traverse record, captured before the session reset, is
        summarised into the planner prompt so the next plan can react to whether
        the last one was adopted and how it actually ran.
        """
        self._pipeline_graph = default_pipeline_graph()
        if not is_auto_planner():
            return
        try:
            planner = self._planner_agent()
            if planner is None:
                self._trace(plan_event(goal=goal, source="static_fallback"))
                return
            history = ""
            try:
                history = summarize_traverse(prev_traverse)
            except Exception:
                history = ""
            prompt = build_plan_prompt(
                from_session(self._session, goal=goal),
                goal,
                history=history,
                lang=self._exec_lang(),
            )
            raw = await planner.complete_text(prompt, max_tokens=1024)
            plan = parse_plan_response(raw)
            if plan is None:
                logger.warning("[Planner] plan generation or validation failed — using the static graph")
                self._trace(plan_event(goal=goal, source="static_fallback"))
                return
            self._pipeline_graph = plan_to_graph(plan)
            self._session["plan"] = plan_to_dict(plan)
            self._trace(plan_event(goal=goal, source="planner", edge_count=len(plan.edges)))
            logger.info(f"[Planner] plan adopted: {len(plan.edges)} edges, goal={goal[:60]!r}")
        except Exception as exc:  # noqa: BLE001 — planning must never kill the pipeline
            logger.warning(f"[Planner] plan build failed, using the static graph: {exc}")
            self._pipeline_graph = default_pipeline_graph()
            self._trace(plan_event(goal=goal, source="static_fallback"))

    # ──────────────────────────────────────────────────────
    # Confirmation detection
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _is_confirmation(message: str, lang: Optional[str] = None) -> bool:
        """Check if the message is a simple confirmation (e.g. "yes", "go ahead", "Option 1")."""
        return _intent_is_confirmation(message, lang)

    @staticmethod
    def _is_skip(message: str, lang: Optional[str] = None) -> bool:
        """Check if the message is a skip or decline (e.g. "no", "skip")."""
        return _intent_is_skip(message, lang)

    # ──────────────────────────────────────────────────────
    # Chain signal detection
    # ──────────────────────────────────────────────────────
    def _detect_chain(self, events: List[AgentEvent]) -> Optional[Dict[str, Any]]:
        """Extract structured chain signal from agent events and resolve next step."""
        for event in events:
            if event.chain_signal is None:
                continue

            stage = event.chain_signal.stage_completed
            metadata = event.chain_signal.metadata

            if stage not in PIPELINE_STAGES:
                continue

            rule = PIPELINE_STAGES[stage]
            metadata = event.chain_signal.metadata

            if stage == "tuning":
                _job = str(metadata.get("task_id") or metadata.get("job_id") or "")
                if _job:
                    self.set_pipeline_train_job(_job)

            # Keep the KBD metadata — the tuning stage reads it.
            if stage == "kbd":
                self._kbd_metadata = {k: metadata[k] for k in _KBD_PRESERVED_KEYS if k in metadata}
                # When KBD started with RAG indexing already complete but the
                # retrieval stage not yet recorded, fill it in.
                if "retrieval" not in [s.get("stage") for s in self._session.get("stages", [])]:
                    try:
                        index_file = CHROMA_DIR / "document_index.json"
                        if index_file.exists():
                            import json as _json

                            docs = _json.loads(index_file.read_text(encoding="utf-8"))
                            if docs:
                                doc_name = docs[0].get("name", "") if isinstance(docs, list) else ""
                                self._record_stage(
                                    "retrieval",
                                    {
                                        "indexed_count": len(docs) if isinstance(docs, list) else 1,
                                        "doc_name": doc_name,
                                    },
                                )
                                if "retrieval" not in self._chain_history:
                                    self._chain_history.append("retrieval")
                    except Exception:
                        pass

            # ── graph mode: transitions come from a conditional edge graph ──
            # next_edge is a 1:1 port of PIPELINE_STAGES / _resolve_kbd_path and
            # picks the next agent; messages reuse the same builders as legacy.
            # On failure it returns None and the legacy path below takes over.
            if is_graph_mode():
                graph_chain = self._graph_transition(stage, metadata)
                if graph_chain is not None:
                    return graph_chain

            # Resolver pattern: dynamic branch (e.g. kbd -> RAG / Corpus / Tuning).
            resolver_fn = rule.get("resolver")
            if resolver_fn and hasattr(self, resolver_fn):
                resolved = getattr(self, resolver_fn)(metadata)
                if resolved is None:
                    continue  # condition not met
                resolved["stage_completed"] = stage
                resolved["metadata"] = metadata
                self._trace(transition_event(src=stage, dst=resolved.get("next_agent"), mode="legacy"))
                return resolved

            # Dynamic message.
            next_message, announce = self._chain_texts(stage, metadata)

            _next = rule["next_agent"]
            result = {
                "stage_completed": stage,
                "next_agent": _next,
                "next_message": next_message,
                "announce": announce,
                "metadata": metadata,
            }
            self._trace(transition_event(src=stage, dst=_next, mode="legacy"))
            return result

        return None

    @staticmethod
    def _orchestration_mode() -> str:
        return "graph" if is_graph_mode() else "legacy"

    def _chain_texts(self, stage: str, metadata: dict) -> "tuple[str, str]":
        """(next_message, announce) for a static transition out of ``stage``.

        Shared by the legacy chain and the graph transition so both modes hand
        the next agent the same instruction.
        """
        if stage == "retrieval":
            return _handoff("retrieval_next", "en"), _handoff("retrieval_announce", "en")
        if stage == "tuning":
            return _handoff("tuning_next", "en", metadata.get("task_id", "")), _handoff("tuning_announce", "en")
        rule = PIPELINE_STAGES.get(stage, {})
        return rule.get("next_message", ""), rule.get("announce", "")

    def _graph_transition(self, stage: str, metadata: dict) -> Optional[Dict[str, Any]]:
        """Graph-mode transition — the conditional edge graph picks the next agent.

        It is a 1:1 port of PIPELINE_STAGES + _resolve_kbd_path, so the decision
        matches legacy, and the messages come from the same builders
        (_chain_texts / _kbd_chain_content). On an exception or an unregistered
        agent it returns None so the caller falls back to legacy.
        """
        try:
            bb = from_session(self._session)
            edge = self._pipeline_graph.next_edge(bb, stage, metadata)
        except Exception as exc:  # noqa: BLE001 — a transition must never die
            logger.warning(f"[Graph] transition failed, falling back to legacy: {exc}")
            return None
        if not edge_is_wireable(edge, self._agents.keys()):
            logger.warning(f"[Graph] cannot wire transition (stage={stage}, edge={edge}), falling back to legacy")
            return None

        if stage == "kbd":
            chain = self._kbd_chain_content(edge.dst, metadata)
        else:
            next_message, announce = self._chain_texts(stage, metadata)
            chain = {
                "next_agent": edge.dst,
                "next_message": next_message,
                "announce": announce,
            }
        chain["stage_completed"] = stage
        chain["metadata"] = metadata
        self._trace(
            transition_event(
                src=stage,
                dst=chain.get("next_agent", edge.dst),
                mode="graph",
                announce_key=edge.announce_key,
            )
        )
        logger.info(f"[Graph] transition: {stage} -> {edge.dst} ({edge.announce_key})")
        return chain

    def _fallback_chain_from_tool_results(self, events: List[AgentEvent], agent_name: str) -> Optional[Dict[str, Any]]:
        """Fallback: parse __chain_signal__ out of the raw tool_result JSON.

        The model may append a text reply after a tool call;
    An agent can yield a chain_signal without it being set on the AgentEvent
    object. This recovers the chain by reading __chain_signal__ straight from
    the tool_result text.
        """
        for event in events:
            if event.role != "tool_result" or not event.tool_result:
                continue
            try:
                parsed = json.loads(event.tool_result)
                if not isinstance(parsed, dict) or "__chain_signal__" not in parsed:
                    continue
                sig = parsed["__chain_signal__"]
                stage = sig.get("stage_completed", "")
                metadata = sig.get("metadata", {})
                if not stage or stage not in PIPELINE_STAGES:
                    continue
                # Restore the chain_signal object and handle it like _detect_chain.
                event.chain_signal = ChainSignal(stage_completed=stage, metadata=metadata)
                logger.debug(
                    f"[DISPATCHER] Fallback chain recovered from tool_result: stage={stage}, agent={agent_name}"
                )
                # Re-enter _detect_chain.
                return self._detect_chain(events)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return None

    def _save_stage_summary(self, chain: Dict[str, Any]) -> None:
        """Store the summary metadata for a stage when a chain is detected."""
        import time as _time

        stage = chain.get("stage_completed", "")
        metadata = chain.get("metadata", {})

        # Per-stage summary.
        summary: Dict[str, Any] = {
            "completed_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "next_agent": chain.get("next_agent"),
            "announce": chain.get("announce", ""),
        }

        if stage == "retrieval":
            summary["description"] = "RAG indexing complete"
            summary["detail"] = metadata.get("doc_name", "")
        elif stage == "kbd":
            coverage = metadata.get("knowledge_coverage_pct", 0)
            rec = metadata.get("recommendation", "")
            summary["description"] = f"KBD analysis — coverage {coverage}%"
            summary["detail"] = f"verdict: {rec}"
            summary["coverage_pct"] = coverage
            summary["recommendation"] = rec
            summary["weak_categories"] = metadata.get("weak_categories", [])
            summary["boundary_categories"] = metadata.get("boundary_categories", [])
            summary["hallucination_rate_pct"] = metadata.get("hallucination_rate_pct", 0)
        elif stage == "tuning":
            summary["description"] = "Training complete"
            summary["detail"] = metadata.get("task_id", "")
            summary["model_name"] = metadata.get("model_name", "")

        self._stage_summaries[stage] = summary
        self._record_stage(stage, metadata)

    def _get_pipeline_config(self) -> dict:
        """Merged system_config + user_config for an AUTO pipeline run."""
        from modules.agents.foundation import AgentBase

        config = AgentBase._load_system_config()

        user_cfg_path = self._storage_root / "user_config.json"
        if user_cfg_path.exists():
            try:
                user_cfg = json.loads(user_cfg_path.read_text(encoding="utf-8"))
                config.update(user_cfg)
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "pipeline_goal": config.get("pipeline_goal", ""),
            "dlmax_engine": config.get("dlmax_engine", "algo"),
            _LANG_CONFIG_KEY: config.get(_LANG_CONFIG_KEY),
        }

    # ──────────────────────────────────────────────────────
    # Condition functions
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _needs_finetuning(metadata: dict) -> bool:
        """Boundary → Corpus chain only fires if KBD recommended FT or Hybrid."""
        rec = metadata.get("recommendation", "").lower()
        return any(kw in rec for kw in _FT_KEYWORDS)

    async def _emit_skip_to_tuning_card(self, land_chain: Dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        """Emit the recommendation card when skipping lands on the training stage.

        Updates land_chain.next_message in place with the recommended parameters
        and yields a "proceed" confirmation card, matching the skip-to-tuning flow.
        """
        ctx = self._gather_training_context(land_chain.get("metadata", {}))
        _rec = await self._build_training_recommendation_llm(land_chain.get("metadata", {}))
        yield AgentEvent(
            role="assistant",
            content=_rec,
            status=AgentPhase.RESPONDING,
            agent_name="tuning",
        )
        land_chain["next_message"] = (
            f"{_ct(self._exec_lang())['confirmed_below_recommended']}\n"
            f"dataset='{ctx['dataset_name']}', job_name='{ctx['dataset_name']}_training'\n\n"
            f"{_ct(self._exec_lang())['recommendation_body']}\n{_rec}\n\n"
            f"{_ct(self._exec_lang())['use_recommendation_as_is']}"
        )
        _announce = _rt("announce_training_params", self._exec_lang())
        yield AgentEvent(
            role="system",
            content=(
                _rt("next_step_heading", self._exec_lang(), announce=_announce)
                + _confirm_hint(self._exec_lang())
                + f"[chain_target:{land_chain['next_agent']}]"
            ),
            status=AgentPhase.RESPONDING,
            agent_name="dispatcher",
        )

    def _resolve_kbd_path(self, metadata: dict) -> Optional[Dict[str, Any]]:
        """Dynamic branch after KBD: RAG, Corpus (data generation) or Tuning."""
        if not self._needs_finetuning(metadata):
            next_agent = None  # RAG verdict — no chaining, guidance only
        else:
            next_agent = "tuning"
        return self._kbd_chain_content(next_agent, metadata)

    def _kbd_chain_content(self, next_agent: Optional[str], metadata: dict) -> Dict[str, Any]:
        """Render the guidance message for each KBD branch.

        Kept separate from the branch *decision* so the legacy path
        (_resolve_kbd_path) and the graph path share identical wording.
        """
        _lang = self._exec_lang()
        coverage = metadata.get("knowledge_coverage_pct", 0)

        if next_agent is None:
            _kbd_model = metadata.get("model_name", "") or self._get_kbd_model()
            if _kbd_model:
                self.set_train_identity(base_model=_kbd_model)
            return {
                "next_agent": None,
                "next_message": _handoff("kbd_rag_next", _lang, coverage, _kbd_model),
                "announce": _handoff("kbd_rag_announce", _lang, coverage),
            }

        if next_agent == "tuning":
            # Fine-tuning or hybrid with training data available -> Tuning
            _kbd_model = metadata.get("model_name", "") or self._get_kbd_model()
            return {
                "next_agent": "tuning",
                "next_message": _handoff("kbd_tuning_next", _lang, _kbd_model),
                "announce": _handoff("kbd_tuning_announce", _lang, coverage),
            }

        # FT/Hybrid but no training dataset yet → ask the user to upload one
        return {
            "next_agent": None,
            "next_message": _handoff("kbd_upload_next", _lang, coverage),
            "announce": _handoff("kbd_upload_announce", _lang, coverage),
        }

    def _get_kbd_model(self) -> str:
        """Model name used in the KBD stage, read from the session."""
        for s in self._session.get("stages", []):
            if s.get("stage") == "kbd" and s.get("model_name"):
                return s["model_name"]
        # _kbd_metadata survives the session reset in start_auto_pipeline.
        return (self._kbd_metadata or {}).get("model_name", "")

    def _list_training_datasets(self) -> list:
        import json as _json

        # Datasets: confirmed training data (corpus) plus unconfirmed output
        # (generated_corpus), so data is found even when the user skipped ahead.
        datasets = []
        for base_dir, source_tag in zip(dataset_roots(), ("training", "generated")):
            if not base_dir.exists():
                continue
            for d in base_dir.iterdir():
                if not d.is_dir():
                    continue
                all_names = [f.name for f in (list(d.glob("*.json")) + list(d.glob("*.jsonl")))]
                selected = select_data_files(all_names)
                if not selected:
                    continue
                count = 0
                for name in selected:
                    f = d / name
                    try:
                        if f.suffix == ".jsonl":
                            count += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
                        else:
                            count += len(_json.loads(f.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                datasets.append(
                    {
                        "name": d.name,
                        "count": count,
                        "is_twist": "_twist" in d.name,
                        "source": source_tag,
                    }
                )
        return datasets

    @staticmethod
    def _installed_model_names() -> List[str]:
        """Base models on disk, by folder name."""
        models_dir = STORAGE_ROOT / "models"
        if not models_dir.exists():
            return []
        return [d.name for d in sorted(models_dir.iterdir()) if d.is_dir() and (d / "config.json").exists()]

    def _gather_training_context(self, metadata: dict) -> dict:
        """Collect training-related information from disk and the API."""
        import json as _json

        # Find models.
        models_dir = STORAGE_ROOT / "models"
        base_models = []
        if models_dir.exists():
            for d in models_dir.iterdir():
                if d.is_dir() and (d / "config.json").exists():
                    # Try to read the model size from config.json.
                    size_info = ""
                    try:
                        cfg = _json.loads((d / "config.json").read_text(encoding="utf-8"))
                        params = cfg.get("num_parameters") or cfg.get("n_params")
                        if params:
                            size_info = f" ({params / 1e9:.1f}B)" if params > 1e6 else ""
                    except Exception:
                        pass
                    base_models.append({"name": d.name, "size": size_info})

        gpu_info = ""
        try:
            from server.core import accelerator

            gpu_info = accelerator.inventory_summary()
        except Exception:
            pass

        # KBD model.
        kbd_model = self._get_kbd_model()

        datasets = self._list_training_datasets()

        # Dataset chosen in the metadata.
        ds_name = metadata.get("job_name", "")
        if ds_name:
            corpus_names = {ds["name"] for ds in datasets if ds["source"] == "training"}
            ds_names_set = {ds["name"] for ds in datasets}
        # An augmented copy wins over the original.
            if ds_name not in corpus_names and f"{ds_name}_twist" in corpus_names:
                ds_name = f"{ds_name}_twist"
            elif ds_name not in ds_names_set and f"{ds_name}_twist" in ds_names_set:
                ds_name = f"{ds_name}_twist"
        # No arbitrary default. KBD on one document says nothing about an
        # unrelated dataset that happens to sit first in the list, and a
        # pre-filled confirm card is how that gets trained by accident.

        ds_count = 0
        for ds in datasets:
            if ds["name"] == ds_name:
                ds_count = ds["count"]
                break

        # Rule-based parameter choice. A model named in this request outranks
        # the KBD model, which otherwise persists from an earlier analysis and
        # overrides what the user just asked for.
        _models = [m["name"] for m in base_models]
        _requested = metadata.get("requested_model") or None
        if _requested and _requested not in _models:
            _requested = None
        rec_model = _requested or (kbd_model if kbd_model and kbd_model in _models else (_models[0] if _models else ""))

        # Dataset size decides the default epochs, batch size and learning rate.
        # The backend and the UI modal only understand "sft" and "lora"
        # ("sft" is what the UI labels Full Fine-Tuning). Sending "fft"
        # silently degrades to LoRA in both the modal and the router.
        if ds_count < 50:
            rec_ep, rec_bs, rec_lr, _size_method = 8, 1, "5e-5", "sft"
        elif ds_count < 500:
            rec_ep, rec_bs, rec_lr, _size_method = 5, 4, "2e-5", "sft"
        else:
            rec_ep, rec_bs, rec_lr, _size_method = 3, 4, "3e-5", "lora"

        # The KBD verdict outranks dataset size. Dropping to LoRA just because
        # the dataset is large would contradict a "knowledge very low, needs
        # full FT" verdict. Order: recommendation, then coverage, then size.
        kbd_meta = self._kbd_metadata or {}
        kbd_rec = (kbd_meta.get("recommendation") or "").lower()
        try:
            kbd_cov = float(kbd_meta.get("knowledge_coverage_pct", 100))
        except (TypeError, ValueError):
            kbd_cov = 100.0

        if "fine-tuning" in kbd_rec and "hybrid" not in kbd_rec:
            # Pure fine-tuning: very low coverage means full FT, otherwise by dataset size.
            rec_method = "sft" if kbd_cov < _LOW_COVERAGE_PCT else _size_method
        elif "hybrid" in kbd_rec:
            # Hybrid: a light LoRA.
            rec_method = "lora"
        elif kbd_cov and kbd_cov < _LOW_COVERAGE_PCT:
            # Even with no verdict text, very low coverage means full FT.
            rec_method = "sft"
        else:
            rec_method = _size_method

        # Clamp the recommended batch with the same formula the guard uses, so a
        # value the guard would later cut is never recommended. Based on the
        # device with the most free memory; unknown values keep the recommendation.
        safe_bs = None
        try:
            from server.core.gpu_admission import _gpu_free_mb_by_index
            from server.core.train_batch_guard import safe_batch_for_model

            _free_map = _gpu_free_mb_by_index()
            if _free_map and rec_model:
                safe_bs = safe_batch_for_model(
                    STORAGE_ROOT / "models" / rec_model,
                    rec_method,
                    _DEFAULT_MAX_SEQ_LEN,
                    max(_free_map.values()),
                )
                if safe_bs is not None and safe_bs >= 1:
                    rec_bs = min(rec_bs, safe_bs)
        except Exception:
            safe_bs = None

        return {
            "base_models": base_models,
            "gpu_info": gpu_info,
            "kbd_model": kbd_model,
            "dataset_name": ds_name,
            "dataset_count": ds_count,
            "all_datasets": datasets,
            "rec_model": rec_model,
            "requested_model": _requested,
            "rec_method": rec_method,
            "rec_epochs": rec_ep,
            "rec_batch": rec_bs,
            "rec_lr": rec_lr,
            "safe_batch": safe_bs,  # memory-based ceiling, None when unknown
            "kbd_recommendation": kbd_meta.get("recommendation", ""),
            "kbd_coverage_pct": kbd_meta.get("knowledge_coverage_pct"),
        }

    async def _build_training_recommendation_llm(self, metadata: dict, lang: Optional[str] = None) -> str:
        """Ask the LLM to recommend training parameters, with a rule-based fallback.

        ``lang`` is an argument because the caller parses this text back with a
        parser for the same language. If production and parsing each looked the
        language up themselves, a setting change in between would silently break.

        """
        ctx = self._gather_training_context(metadata)
        _lang = lang if lang is not None else self._exec_lang()

        # No API key: rules only, no LLM call.

        prompt = _train_rec_prompt.build_prompt(
            ctx,
            lang=_lang,
            low_coverage_pct=_LOW_COVERAGE_PCT,
            default_max_seq_len=_DEFAULT_MAX_SEQ_LEN,
        )

        try:
            tuning_agent = self._agents.get("tuning")
            if tuning_agent:
                llm_rec = await tuning_agent.complete_text(prompt, max_tokens=JSON_TOKENS)
                llm_rec = llm_rec.strip()
                if llm_rec and len(llm_rec) > 30:
                    logger.debug(f"[DISPATCHER] LLM training recommendation generated ({len(llm_rec)} chars)")
                    return llm_rec
        except Exception as e:
            logger.debug(f"[DISPATCHER] LLM training recommendation failed, using fallback: {e}")

        return self._build_training_recommendation_fallback(ctx, _lang)

    @staticmethod
    def _build_training_recommendation_fallback(ctx: dict, lang: Optional[str] = None) -> str:
        """Rule-based training parameters — the fallback when no LLM is available.

        This card goes through the same parser as the LLM one, so its labels must
        come from ``LABELS``; otherwise a fallback run silently produces defaults.
        Unset ``lang`` means English, matching every existing caller.
        """
        return _train_rec_prompt.fallback_card(ctx, lang)

    def _exec_lang(self) -> str:
        from server.core.lang import resolve_lang

        try:
            return resolve_lang(config=self._get_pipeline_config())
        except Exception:  # noqa: BLE001 — a config read failure must not block the recommendation
            return "en"

    @staticmethod
    def _parse_llm_rec_params(rec_text: str, ctx: dict, lang: Optional[str] = None) -> dict:
        """Extract training parameters from the recommendation text.

        Falls back to the ctx defaults when parsing fails.
        """
        from server.core.rec_params import parse_rec_params

        return parse_rec_params(rec_text, ctx, lang)

    # ──────────────────────────────────────────────────────
    # Core execution
    # ──────────────────────────────────────────────────────
    async def run(
        self,
        message: str,
        agent_name: Optional[str] = None,
        chain_action: Optional[str] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Main entry point. Processes user message.

        Flow:
        1. If there's a pending chain and user confirms → execute the chain
        2. If there's a pending chain but user says something else → clear it, re-route
        3. Normal routing → execute agent → detect chain → auto-execute or save for guided
        """
        import time as _time

        if not agent_name and message:
            self._last_user_message = message
        self._is_busy = True
        self._busy_agent = agent_name or "dispatcher"
        self._busy_since = _time.time()
        try:
            async for event in self._run_inner(message, agent_name, chain_action):
                yield event
        finally:
            self._is_busy = False
            self._busy_agent = None
            self._busy_since = 0

    async def _run_inner(
        self,
        message: str,
        agent_name: Optional[str] = None,
        chain_action: Optional[str] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        _lang = self._exec_lang()

        _act_confirm = chain_action == "approve"
        _act_skip = chain_action == "skip"

        def _confirmed(msg: str) -> bool:
            return _act_confirm or self._is_confirmation(msg, _lang)

        def _skipped(msg: str) -> bool:
            return _act_skip or self._is_skip(msg, _lang)

        # ── Step 1: Check for pending chain confirmation ──
        _pend = self._pending_chain
        _pend_desc = "yes: " + _pend.get("announce", "")[:30] if _pend else "None"
        logger.debug(
            "[DISPATCHER] _run_inner: pending_chain=%s, skip=%s, confirm=%s, action=%s, agent_name=%s, msg=%s",
            _pend_desc,
            _skipped(message),
            _confirmed(message),
            chain_action,
            agent_name,
            message[:40],
        )

        # ── Step 1a: skip one proposed stage and propose the next (producers are blocked) ──
        if self._pending_chain and _skipped(message):
            # The skip behaviour is decided from the proposed stage.
            proposed_agent = self._pending_chain.get("next_agent")
            decision = decide_skip(proposed_agent)

            skipped_chain = self._pending_chain
            self._pending_chain = None
            self._chain_paused = False
            self._resume_override_pending = None
            skipped_announce = skipped_chain.get("announce", "")
            skipped_meta = skipped_chain.get("metadata", {})

            # Record the skipped stage in the session.
            self._record_stage(
                proposed_agent or skipped_chain.get("stage_completed", ""),
                skipped_meta,
                skipped=True,
                skip_reason="user_choice",
            )

            # ── end: nothing left to skip to (an evaluation terminal, say) ──
            if decision["action"] == "end":
                yield AgentEvent(
                    role="system",
                    content=_rt("chain_skipped_end", self._exec_lang(), announce=skipped_announce),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return

            # ── advance: propose the landing-stage card ──
            yield AgentEvent(
                role="system",
                content=_rt("stage_skipped", self._exec_lang(), stage=skipped_announce),
                status=AgentPhase.RESPONDING,
                agent_name="dispatcher",
            )
            _from_stage = (skipped_chain.get("stage_completed", "") or "") + "_skipped"

            if decision["land"] == "tuning":
                # Preprocessing and augmentation skipped, training next — emit the card.
                land_chain = {
                    **_TUNING_SKIP_CHAIN,
                    "next_message": _handoff(_TUNING_SKIP_CHAIN["next_message_key"], self._exec_lang()),
                    "announce": _rt(_TUNING_SKIP_CHAIN["announce_key"], self._exec_lang()),
                    "stage_completed": _from_stage,
                    "metadata": skipped_meta,
                }
                self._pending_chain = land_chain
                async for event in self._emit_skip_to_tuning_card(land_chain):
                    yield event
                return

            return

        # ── Step 1b: confirmation — run the pending chain ──
        # pending_chain survives until an explicit clear (reset or a new chain signal).
        if self._pending_chain and _confirmed(message):
            chain = self._pending_chain
            self._pending_chain = None
            self._chain_paused = False
            self._resume_override_pending = None
            async for event in self._execute_chain(chain, depth=0):
                yield event
            return

        if not agent_name and self._is_chitchat(message):
            logger.debug(f"[DISPATCHER] Chitchat detected, bypassing specialist: {message[:40]!r}")
            yield AgentEvent(
                role="assistant",
                content=self._chitchat_reply(message),
                status=AgentPhase.RESPONDING,
                agent_name="dispatcher",
            )
            return

        _pending_agent = self._pending_chain.get("next_agent") if self._pending_chain else None
        if _pending_agent and not agent_name:
            _pending_label = stage_label(_pending_agent, self._exec_lang())

            # A question is answered, not executed — keep the pending card and the pause.
            if is_question(message, _lang):
                yield AgentEvent(
                    role="system",
                    content=_rt("resume_question", self._exec_lang(), label=_pending_label),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return

            _matched = route_match(message)
            _kind = classify_resume(_matched, _pending_agent, self._resume_override_pending)

            if _kind == "unmatched":
                # Matches no stage — explain and keep the card.
                _lang_now = self._exec_lang()
                _options = " / ".join(stage_label(a, _lang_now) for a in _STAGE_ORDER if a != "retrieval")
                yield AgentEvent(
                    role="system",
                    content=_rt("resume_unmatched", self._exec_lang(), label=_pending_label, options=_options),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return

            if _kind == "mismatched":
                # A different stage from the proposal — ask. Repeating it proceeds.
                self._resume_override_pending = _matched
                _matched_label = stage_label(_matched, self._exec_lang())
                yield AgentEvent(
                    role="system",
                    content=_rt(
                        "resume_mismatched",
                        self._exec_lang(),
                        label=_pending_label,
                        requested=_matched_label,
                    ),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return

            # "matched" or "override" — resume, and lift the pause since this is explicit.
            self._chain_paused = False
            self._resume_override_pending = None
            if _kind == "override":
                # Drop the proposed stage and go to the one the user named.
                self._pending_chain = None
            # On a match keep pending_chain: Step 2 below routes the message to the
            # specialist, and _detect_chain builds the next card when it finishes.

        # ── Step 2: Route to agent ──
        # A pending_chain is not cleared here — the user can ask a follow-up and then confirm.
        # It is only removed by a new chain signal or a reset.
        if agent_name:
            target_name = agent_name
        elif self._is_confirmation(message, _lang) and self._last_agent:
            # Confirmation or option pick — route back to the last agent.
            # An option pick must always be handled in the previous agent's context.
            import re

            is_option_select = bool(re.match(r"^option\s*\d", message.strip().lower()))
            if is_option_select:
                # Option picks always go to _last_agent, never re-routed.
                target_name = self._last_agent
            else:
                # Plain confirmation: only keep the agent if it made a tool call.
                last_agent_obj = self._agents.get(self._last_agent)
                has_recent_tool = False
                if last_agent_obj and last_agent_obj.conversation:
                    for conv in reversed(last_agent_obj.conversation[-4:]):
                        content = conv.get("content", "")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    has_recent_tool = True
                                    break
                        if has_recent_tool:
                            break
                target_name = self._last_agent if has_recent_tool else await self._route_smart(message)
        else:
            target_name = await self._route_smart(message)
        agent = self._agents.get(target_name)

        if agent is None:
            yield AgentEvent(
                role="system",
                content=_rt("agent_not_found_named", self._exec_lang(), agent=target_name),
                status=AgentPhase.ERROR,
                agent_name="dispatcher",
            )
            return

        # Record the last conversational agent and update the busy indicator.
        self._last_agent = target_name
        self._busy_agent = target_name

        if target_name == "tuning" and not has_pipeline_marker(message):
            _mode = self._session.get("mode") or agent.autonomy
            if _mode != "auto":
                _ds_all = self._list_training_datasets()
                _ds_res = resolve_training_dataset(message, _ds_all)
                if _ds_res.status in (STATUS_ABSENT, STATUS_AMBIGUOUS):
                    _ds_key = (
                        "train_dataset_not_found" if _ds_res.status == STATUS_ABSENT else "train_dataset_ambiguous"
                    )
                    _ds_preview = "\n".join(f"- {n}" for n in dataset_names(_ds_all)[:10])
                    logger.warning(
                        "[DISPATCHER] guided training refused (%s): requested target %r is not in the real list or is ambiguous — %r",
                        _ds_res.status,
                        _ds_res.mentioned,
                        message[:120],
                    )
                    yield AgentEvent(
                        role="assistant",
                        content=_rt(
                            _ds_key,
                            self._exec_lang(),
                            mentioned=", ".join(_ds_res.mentioned),
                            preview=_ds_preview,
                        ),
                        status=AgentPhase.RESPONDING,
                        agent_name="tuning",
                    )
                    return
                _ds_pick = _ds_res.name
                _ds_hint: Dict[str, Any] = {}
                if _ds_pick:
                    _ds_hint["job_name"] = _ds_pick
                _model_pick = resolve_requested_model(message, self._installed_model_names())
                if _model_pick:
                    _ds_hint["requested_model"] = _model_pick
                    logger.info("[DISPATCHER] model requested in this message: %s", _model_pick)
                if not _ds_pick:
                    logger.info(
                        "[DISPATCHER] guided training: the message names no dataset — "
                        "the pipeline must supply one, nothing is pre-selected: %r",
                        message[:120],
                    )
                ctx = self._gather_training_context(_ds_hint)
                _rec_lang = self._exec_lang()
                _rec = await self._build_training_recommendation_llm(_ds_hint, _rec_lang)
                # Recommendation message (icon plus bullet list).
                yield AgentEvent(
                    role="assistant",
                    content=_rec,
                    status=AgentPhase.RESPONDING,
                    agent_name="tuning",
                )
                # Parse the parameters from the LLM text so the modal and next_message agree.
                _p = self._parse_llm_rec_params(_rec, ctx, _rec_lang)
                _p_method = _p["rec_method"]
                _p_model = _p["rec_model"] or ctx.get("rec_model", "")
                _p_dataset = ctx.get("dataset_name", "")
                if not _p_dataset:
                    yield AgentEvent(
                        role="system",
                        content=self._unresolved_dataset_message(ctx),
                        status=AgentPhase.RESPONDING,
                        agent_name="dispatcher",
                    )
                    return
                _p_job = f"{_p_dataset}_training"
                _p_epochs = _p["rec_epochs"]
                _p_batch = _p["rec_batch"]
                _p_lr = _p["rec_lr"]
                _direct_msg = (
                    f"{_ct(self._exec_lang())['confirmed_recommended']}\n"
                    f"{_ct(self._exec_lang())['exact_params_label']}base_model='{_p_model}', "
                    f"dataset='{_p_dataset}', "
                    f"job_name='{_p_job}', "
                    f"method='{_p_method}', "
                    f"epochs={_p_epochs}, "
                    f"batch_size={_p_batch}, "
                    f"learning_rate='{_p_lr}', "
                    f"max_seq_length={_DEFAULT_MAX_SEQ_LEN}"
                )
                self._pending_chain = {
                    "stage_completed": "",
                    "next_agent": "tuning",
                    "next_message": _direct_msg,
                    "announce": _rt("announce_training_params", self._exec_lang()),
                    "metadata": {},
                    "recommended_params": {
                        "stage": "tuning",
                        "method": _p_method,
                        "base_model": _p_model,
                        "base_model_options": [m.get("name", "") for m in ctx.get("base_models", []) if m.get("name")],
                        "dataset": _p_dataset,
                        "dataset_name": _p_dataset,
                        "dataset_options": [d.get("name", "") for d in ctx.get("all_datasets", []) if d.get("name")],
                        "job_name": _p_job,
                        "epochs": _p_epochs,
                        "batch_size": _p_batch,
                        "learning_rate": _p_lr,
                        "max_seq_length": _DEFAULT_MAX_SEQ_LEN,
                        "lora_r": _DEFAULT_LORA_R,
                        "lora_alpha": _DEFAULT_LORA_ALPHA,
                        "lora_dropout": _DEFAULT_LORA_DROPOUT,
                    },
                }
                _announce_tuning = _rt("announce_training_params", self._exec_lang())
                yield AgentEvent(
                    role="system",
                    content=(
                        _rt("next_step_heading", self._exec_lang(), announce=_announce_tuning)
                        + _confirm_hint(self._exec_lang())
                        + "[chain_target:tuning]"
                    ),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return
            else:
                _kbd_model = self._get_kbd_model()
                _model_hint = f"Base model: {_kbd_model}\n" if _kbd_model else ""
                message += (
                    f"\n\n{AUTO_MARKER} {_model_hint}Call the appropriate tool and start training now, without asking again."
                )

        # ── Step 4: Execute agent and collect events ──
        collected_events: List[AgentEvent] = []
        async for msg in agent.run(message):
            yield msg
            collected_events.append(msg)

        # ── Step 5: Detect chain signal ──
        chain_events = [e for e in collected_events if e.chain_signal is not None]
        logger.debug(
            f"[DISPATCHER] Agent '{target_name}' produced {len(collected_events)} events, {len(chain_events)} with chain signals"
        )
        for ce in chain_events:
            logger.debug(
                f"[DISPATCHER]   chain_signal: stage={ce.chain_signal.stage_completed}, metadata_keys={list(ce.chain_signal.metadata.keys())}"
            )
        chain = self._detect_chain(collected_events)
        # Fallback: no chain_signal, so parse __chain_signal__ out of the tool_result JSON.
        if chain is None:
            chain = self._fallback_chain_from_tool_results(collected_events, target_name)
        if chain is None:
            logger.debug("[DISPATCHER] No valid chain detected")
            had_tool_calls = any(e.role == "tool_result" for e in collected_events)
            if agent.autonomy == "guided" and had_tool_calls and target_name not in self._chain_history:
                self._chain_history.append(target_name)
                self._record_stage(target_name, {})
            return
        logger.debug(
            f"[DISPATCHER] Chain detected: stage={chain.get('stage_completed')}, next={chain.get('next_agent')}, announce={chain.get('announce', '')[:50]}"
        )
        self._save_stage_summary(chain)

        # ── Step 6: Auto vs Guided ──
        # A next_agent of None means guidance only (an RAG verdict, for example).
        if chain.get("next_agent") is None:
            if chain.get("announce"):
                yield AgentEvent(
                    role="system",
                    content=chain["announce"],
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
            return

        autonomy = agent.autonomy
        # An auto session forces auto autonomy (guards against a failed config read).
        if self._session.get("mode") == "auto" and autonomy != "auto":
            autonomy = "auto"
        # A guided session forces guided autonomy.
        if self._session.get("mode") == "guided" and autonomy != "guided":
            autonomy = "guided"
        # While chaining is paused, auto waits for confirmation like guided.
        if self._chain_paused and autonomy == "auto":
            autonomy = "guided"
        logger.debug(
            f"[DISPATCHER] Step 6: autonomy={autonomy}, session_mode={self._session.get('mode')}, paused={self._chain_paused}, chain_announce={chain.get('announce', '')[:50]}"
        )
        # ── Guided: stage completion summary, shown above the chain card ──
        if autonomy == "guided":
            summary = self._build_stage_summary(chain)
            if summary:
                yield AgentEvent(
                    role="assistant",
                    content=summary,
                    status=AgentPhase.RESPONDING,
                    agent_name=target_name,
                )

        if autonomy == "auto":
            async for event in self._execute_chain(chain, depth=0, pipeline_mode="auto"):
                yield event
        else:
            logger.debug("[DISPATCHER] Guided mode -- saving pending_chain and yielding card")
            self._pending_chain = chain
            async for event in self._enrich_and_announce_pending_chain(chain):
                yield event

    def _unresolved_dataset_message(self, ctx: Dict[str, Any]) -> str:
        """What to say when nothing tied a dataset to this run.

        Naming the datasets that do exist matters: the failure is "which one",
        not "there are none", and the two need different answers from the user.
        """
        known = [d.get("name", "") for d in ctx.get("all_datasets", []) if d.get("name")]
        logger.warning(
            "[DISPATCHER] no dataset resolved for this run — confirm card withheld (%d known)",
            len(known),
        )
        if known:
            return _rt(
                "train_dataset_unresolved",
                self._exec_lang(),
                preview="\n".join(f"- {n}" for n in known[:10]),
            )
        return _rt("res_no_training_dataset", self._exec_lang())

    async def _enrich_and_announce_pending_chain(self, chain: Dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        """Store the pending chain, then yield, in order: the per-stage LLM
        recommendation card; a baked ``next_message`` plus ``recommended_params``
        snapshot the user can edit; and the final ``[chain_target:...]`` card.

        Both the top-level chaining path and the ``_execute_chain`` recursion
        (the gate after a stage skipped with inline_continue) must call this.
        Calling it from only one leaves pending_chain_params empty, and the UI
        jumps straight to advanceChain without opening the edit modal.
        """
        next_agent_name = chain.get("next_agent")
        metadata = chain.get("metadata", {}) or {}
        announce = chain.get("announce", "")

        if next_agent_name == "tuning":
            ctx = self._gather_training_context(metadata)
            _rec_lang = self._exec_lang()
            rec = await self._build_training_recommendation_llm(metadata, _rec_lang)
            yield AgentEvent(
                role="assistant",
                content=rec,
                status=AgentPhase.RESPONDING,
                agent_name="tuning",
            )
            # Parse the parameters from the LLM text so the modal, next_message and card agree.
            dataset = ctx.get("dataset_name", "")
            if not dataset:
                # The card carries a "start training?" button, so it must not
                # open on a dataset nothing selected.
                self._pending_chain = None
                yield AgentEvent(
                    role="system",
                    content=self._unresolved_dataset_message(ctx),
                    status=AgentPhase.RESPONDING,
                    agent_name="dispatcher",
                )
                return
            _parsed = self._parse_llm_rec_params(rec, ctx, _rec_lang)
            rec_method = _parsed["rec_method"]
            rec_model = _parsed["rec_model"] or ctx.get("rec_model", "")
            job_name = f"{dataset}_training"
            rec_epochs = _parsed["rec_epochs"]
            rec_batch = _parsed["rec_batch"]
            rec_lr = _parsed["rec_lr"]
            _pcfg = self._get_pipeline_config()
            _goal = _pcfg["pipeline_goal"]
            max_seq_length = _DEFAULT_MAX_SEQ_LEN

            lora_bits = ""
            lora_r = _DEFAULT_LORA_R
            lora_alpha = _DEFAULT_LORA_ALPHA
            lora_dropout = _DEFAULT_LORA_DROPOUT
            if rec_method == "lora":
                lora_bits = f", lora_r={lora_r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}"

            _goal_ctx = _ct(self._exec_lang())["goal_context"].format(value=_goal) if _goal else ""
            chain["next_message"] = (
                f"{_ct(self._exec_lang())['confirmed_below']}{_goal_ctx}\n"
                f"{_ct(self._exec_lang())['exact_params_label']}base_model='{rec_model}', "
                f"dataset='{dataset}', "
                f"job_name='{job_name}', "
                f"method='{rec_method}', "
                f"epochs={rec_epochs}, "
                f"batch_size={rec_batch}, "
                f"learning_rate='{rec_lr}', "
                f"max_seq_length={max_seq_length}"
                f"{lora_bits}"
            )
            chain["recommended_params"] = {
                "stage": "tuning",
                "method": rec_method,
                "base_model": rec_model,
                "base_model_options": [m.get("name", "") for m in ctx.get("base_models", []) if m.get("name")],
                "dataset": dataset,
                "dataset_name": dataset,
                "dataset_options": [d.get("name", "") for d in ctx.get("all_datasets", []) if d.get("name")],
                "job_name": job_name,
                "epochs": rec_epochs,
                "batch_size": rec_batch,
                "learning_rate": rec_lr,
                "max_seq_length": max_seq_length,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            }
            announce = _rt("announce_training_params", self._exec_lang())

        yield AgentEvent(
            role="system",
            content=(
                _rt("next_step_heading", self._exec_lang(), announce=announce)
                + _confirm_hint(self._exec_lang())
                + f"[chain_target:{next_agent_name}]"
            ),
            status=AgentPhase.RESPONDING,
            agent_name="dispatcher",
        )

    async def _execute_chain(
        self,
        chain: Dict[str, Any],
        depth: int,
        pipeline_mode: Optional[str] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Execute a chain step, then recursively check for further chaining.

        Auto mode chains continue recursively up to _MAX_CHAIN_DEPTH.
        Guided mode saves the next chain for user confirmation and stops.

        pipeline_mode: "auto" forces every chained stage to run in auto.
        """
        if depth >= _MAX_CHAIN_DEPTH:
            yield AgentEvent(
                role="system",
                content=_rt("max_depth_reached", self._exec_lang()),
                status=AgentPhase.RESPONDING,
                agent_name="dispatcher",
            )
            return

        next_agent = self._agents.get(chain["next_agent"])
        if next_agent is None:
            return

        self._trace(execute_event(agent=chain["next_agent"], src=chain.get("stage_completed", "")))

        # Auto pipeline: the next agent switches to auto as well.
        if pipeline_mode:
            next_agent.set_autonomy(pipeline_mode)

        # Update pipeline state + last agent tracking + busy agent
        stage_completed = chain.get("stage_completed", "")
        if stage_completed and stage_completed not in self._chain_history:
            self._chain_history.append(stage_completed)
        self._current_stage = chain["next_agent"]
        self._last_agent = chain["next_agent"]  # the chained agent is now current
        self._busy_agent = chain["next_agent"]  # shown as running in the UI

        # Announce only the next step to the user, plus the page-navigation marker.
        yield AgentEvent(
            role="system",
            content=f"{chain['announce']}\n[navigate:{chain['next_agent']}]",
            status=AgentPhase.THINKING,
            agent_name="dispatcher",
        )

        # Build message — enrich with metadata from completed stage
        message = chain["next_message"]
        metadata = chain.get("metadata", {})

        # KBD metadata: from the current chain if present, else the preserved copy.
        kbd = metadata if metadata.get("recommendation") else self._kbd_metadata
        if kbd:
            weak = kbd.get("weak_categories", [])
            boundary = kbd.get("boundary_categories", [])
            coverage = kbd.get("knowledge_coverage_pct", "N/A")
            hall = kbd.get("hallucination_rate_pct", "N/A")
            rec = kbd.get("recommendation", "N/A")
            message += (
                f"\n\n[KBD analysis]\n"
                f"- Verdict: {rec}\n"
                f"- Knowledge coverage: {coverage}%\n"
                f"- Hallucination rate: {hall}%\n"
                f"- Weak categories (outside): {', '.join(weak) if weak else 'none'}\n"
                f"- Boundary categories: {', '.join(boundary) if boundary else 'none'}\n"
                f"Focus the work on these categories."
            )

        # The dispatcher already has the user's confirmation (GuidedChainCard), so
        # the agent normally switches to auto and runs without asking again.
        # Exception: the assessment stage of a guided session stays guided, so the
        # evaluation tools hit their guard and open the modal that asks which
        # benchmark and how many queries — confirming a stage is not the same as
        # choosing evaluation parameters. An auto pipeline stays fully automatic.
        next_agent.set_autonomy("auto")
        message += (
            f"\n\n{AUTO_MARKER} "
            "The user has already confirmed this pipeline step. "
            "Call the appropriate tool and run it now, without asking again. "
            "Do not ask questions such as 'shall I proceed?'."
        )

        # Execute next agent
        collected_events: List[AgentEvent] = []
        try:
            async for msg in next_agent.run(message):
                yield msg
                collected_events.append(msg)
        finally:
            # Drop the autonomy override and restore the original setting.
            next_agent.clear_autonomy_override()

        # Recursive: check if this agent also produced a chain signal
        next_chain = self._detect_chain(collected_events)
        if next_chain is None:
            next_chain = self._fallback_chain_from_tool_results(collected_events, chain["next_agent"])
        if next_chain is None:
            return
        self._save_stage_summary(next_chain)

        # ``inline_continue``: when the user approved a higher-level step (such as
        # preprocessing and augmentation), its internal sub-steps are not confirmed
        # again. The flag bypasses exactly one gate — the transition after it
        # (twist to tuning, say) must hit the guided gate again.
        # pipeline_mode is deliberately not propagated: "auto" there would run
        # every sub-step without a gate.
        rule_for_completed = PIPELINE_STAGES.get(next_chain.get("stage_completed", ""), {})
        inline_continue = bool(rule_for_completed.get("inline_continue"))

        autonomy = pipeline_mode or self._session.get("mode") or next_agent.autonomy

        if autonomy == "auto" or inline_continue:
            async for event in self._execute_chain(
                next_chain,
                depth=depth + 1,
                # inline_continue is a one-step bypass, so it must not make
                # pipeline_mode sticky at "auto". An outer auto run is inherited
                # as it is.
                pipeline_mode=pipeline_mode,
            ):
                yield event
        else:
            # Guided: save for user confirmation and stop.
            # The gate reached after an inline_continue skip needs the same
            # enrichment (recommendation card, recommended_params, baked
            # next_message) or the UI edit modal will not open.
            self._pending_chain = next_chain
            async for event in self._enrich_and_announce_pending_chain(next_chain):
                yield event

    # ──────────────────────────────────────────────────────
    # Stage summary shown in guided and manual mode
    # ──────────────────────────────────────────────────────
    def _build_stage_summary(self, chain: Dict[str, Any]) -> str:
        """Build the stage-completion summary from the chain signal metadata.

        The ``rec`` values (``RAG`` / ``Fine-Tuning`` / ``Hybrid``) are the machine
        contract set by ``kbd_verdict`` — never translate them.

        """
        stage = chain.get("stage_completed", "")
        metadata = chain.get("metadata", {})
        if not metadata:
            return ""

        _lang = self._exec_lang()

        if stage == "kbd":
            coverage = metadata.get("knowledge_coverage_pct", "N/A")
            rec = metadata.get("recommendation", "N/A")
            hall = metadata.get("hallucination_rate_pct", "N/A")
            weak = metadata.get("weak_categories", [])
            boundary = metadata.get("boundary_categories", [])
            # Verdict text. The branch uses machine values; only wording is localised.
            if rec == "RAG":
                verdict = _rt("stage_kbd_verdict_rag", _lang)
            elif rec == "Fine-Tuning":
                if isinstance(coverage, (int, float)) and coverage < _LOW_COVERAGE_PCT:
                    verdict = _rt("stage_kbd_verdict_ft_low", _lang)
                else:
                    verdict = _rt("stage_kbd_verdict_ft", _lang)
            elif rec == "Hybrid":
                verdict = _rt("stage_kbd_verdict_hybrid", _lang)
            else:
                verdict = _rt("stage_kbd_verdict_other", _lang, value=rec)
            lines = [
                _rt("stage_kbd_heading", _lang),
                _rt("stage_kbd_coverage", _lang, value=coverage),
                _rt("stage_kbd_hallucination", _lang, value=hall),
                _rt("stage_kbd_verdict_row", _lang, value=rec),
            ]
            if weak:
                lines.append(_rt("stage_kbd_weak", _lang, categories=", ".join(weak[:5])))
            if boundary:
                lines.append(_rt("stage_kbd_boundary", _lang, categories=", ".join(boundary[:5])))
            lines.append(f"\n{verdict}")
            return "\n".join(lines)

        # The UI groups preprocessing and twist into one "preprocess and augment"
        # step, so the preprocessing summary is omitted and only twist emits the
        # combined summary.

        if stage == "tuning":
            model = metadata.get("model_name", "")
            method = metadata.get("method", "")
            return _rt("stage_tuning_done", _lang, model=model, method=method) if model else ""

        return ""
