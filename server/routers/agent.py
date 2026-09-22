"""Agent router — /api/agent

GET  /status            — agent system status
GET  /orchestration/trace — read model of the traverse record
POST /reset             — reset agent state
POST /dispatch          — SSE dispatcher stream (Guided mode)
GET  /pipeline          — dispatcher pipeline state (Guided mode)
POST /chain/advance     — approve/skip the pending chain (Guided mode)
POST /chain/pause       — pause auto-chaining
POST /chain/resume      — resume auto-chaining
"""

import asyncio
import json
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.core.lang import resolve_lang
from server.core.logging import sys_log
from server.core.pipeline_config import load_pipeline_config

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# Per-user dispatcher instances (lazy init)
_dispatchers: Dict[str, object] = {}


def _get_dispatcher(user_id: str):
    """Get or create an AgentDispatcher for the user."""
    if user_id not in _dispatchers:
        from modules.agents.dispatcher import AgentDispatcher

        _dispatchers[user_id] = AgentDispatcher(user_id=user_id)
    return _dispatchers[user_id]


async def _save_chat_message(
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    *,
    tool_use: list | None = None,
    thinking: str | None = None,
    agent_name: str | None = None,
) -> None:
    from server.core.chat_save import save_chat_message

    await save_chat_message(
        thread_id,
        user_id,
        role,
        content,
        tool_use=tool_use,
        thinking=thinking,
        agent_name=agent_name,
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    user_id: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------


@router.get("/status")
async def agent_status(request: Request):
    """Agent system status."""
    user_id = getattr(request.state, "user_id", "default")
    try:
        dispatcher = _get_dispatcher(user_id)
        agents = dispatcher.list_agents()
        return {
            "status": "active",
            "user_id": user_id,
            "agents": agents,
            "is_busy": getattr(dispatcher, "_is_busy", False),
            "busy_agent": getattr(dispatcher, "_busy_agent", None),
            "current_stage": getattr(dispatcher, "_current_stage", None),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/orchestration/trace")
async def orchestration_trace(request: Request):
    """Orchestration traverse record for this session (routing, transition and

    The event schema is in server/core/orchestration/trace.py. Sessions reset
    on server restart, so this record is per session.
    """
    from server.core.orchestration import TRACE_KEY, orchestrator_mode

    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)
    session = getattr(dispatcher, "_session", None) or {}
    return {
        "session_id": session.get("session_id", ""),
        "session_mode": session.get("mode", ""),
        "orchestrator_mode": orchestrator_mode(),
        "events": list(session.get(TRACE_KEY, [])),
    }


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------


@router.post("/reset")
async def reset_agent(req: ResetRequest, request: Request):
    """Reset agent state for the user."""
    user_id = req.user_id or getattr(request.state, "user_id", "default")

    # Remove cached dispatcher to force re-initialization
    if user_id in _dispatchers:
        del _dispatchers[user_id]

    sys_log(f"[Agent] Agent state reset (user={user_id})")
    return {"message": "Agent state reset", "user_id": user_id}


# ---------------------------------------------------------------------------
# POST /ask — REST chat endpoint (SSE stream)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Guided-mode dispatcher endpoints
# ---------------------------------------------------------------------------
#
# These endpoints expose the AgentDispatcher (which owns pipeline chaining +
# guided/auto logic) to the frontend as SSE streams.
#
# Event types emitted by /dispatch:
#   {type: "event", ...AgentEvent.to_dict()}
#     Regular agent stream (thinking / tool_use / tool_result / assistant / system).
#     System events may embed markers in `content`:
#       [chain_target:<agent>]  — a guided chain card is pending
#       [navigate:<agent>]      — UI should navigate to that page
#       [stage_complete:<stage>] — a pipeline stage has finished
#   {type: "status", ...pipeline_status}
#     Emitted once after the stream finishes. Contains completed_stages,
#     pending_chain, current_stage, chain_paused, is_busy, session.
#   {type: "error", message: string}
#   {type: "done"}


def advance_message(action: str, lang: object = None) -> str:
    """Text for an advance action. Display and record only — the decision
    itself comes from ``chain_action``."""
    from server.core.agent_runtime_texts import text

    return text(f"chain_advance_{action}", lang)


class AgentDispatchRequest(BaseModel):
    message: str
    agent: Optional[str] = None
    thread_id: Optional[str] = None


class ChainAdvanceRequest(BaseModel):
    action: Literal["approve", "skip"]
    thread_id: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


class ExecutionConfirmRequest(BaseModel):
    """`overrides` holds the values the modal edited. **Only allowlisted keys**
    are applied (`pending_execution.EDITABLE_FIELDS`); merging arbitrary keys
    would pass arguments outside the tool contract and break the run.

    """

    action: Literal["approve", "cancel"] = "approve"
    overrides: Optional[dict] = None
    thread_id: Optional[str] = None


class EvalConfirmRequest(BaseModel):
    """- SpectraBench: model_name/profile/tasks

    """

    action: Literal["approve", "cancel"] = "approve"
    model_name: Optional[str] = None
    profile: Optional[str] = None
    tasks: Optional[list] = None
    max_queries: Optional[int] = None
    thread_id: Optional[str] = None


async def _dispatcher_sse(
    dispatcher,
    message: str,
    agent_name: Optional[str],
    chain_action: Optional[str] = None,
):
    """Shared SSE generator: runs dispatcher.run() and emits structured events.

    Emits a `: keepalive` SSE comment every ~15s while waiting for the next
    event so the connection (through proxies / browser ReadableStream) doesn't
    die during long-running tool executions (e.g. corpus specialist polling
    for QA generation completion can silently wait 5-20 minutes).
    """
    gen = dispatcher.run(message=message, agent_name=agent_name, chain_action=chain_action)
    KEEPALIVE_INTERVAL = 15.0
    try:
        while True:
            next_evt = asyncio.ensure_future(gen.__anext__())
            try:
                while not next_evt.done():
                    # Wait for either the next event or the keepalive tick
                    await asyncio.wait({next_evt}, timeout=KEEPALIVE_INTERVAL)
                    if not next_evt.done():
                        # Still waiting — send a comment frame to keep the
                        # connection alive. SSE comments are `:`-prefixed lines.
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                next_evt.cancel()
                raise

            try:
                event = next_evt.result()
            except StopAsyncIteration:
                break

            try:
                payload = {"type": "event", **event.to_dict()}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as e:
                sys_log(f"[Agent] Dispatch serialization error: {e}", level="WARNING")
                continue
    except asyncio.CancelledError:
        raise
    except Exception as e:
        sys_log(f"[Agent] Dispatch runtime error: {e}", level="ERROR")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    # Final status snapshot — frontend uses this to render chain card
    try:
        status = dispatcher.get_pipeline_status()
        yield f"data: {json.dumps({'type': 'status', **status}, ensure_ascii=False, default=str)}\n\n"
    except Exception as e:
        sys_log(f"[Agent] Dispatch status error: {e}", level="WARNING")

    yield 'data: {"type": "done"}\n\n'


async def _save_agent_message(
    user_id: str,
    role: str,
    content: str,
    *,
    agent_name: str | None = None,
    thinking: str | None = None,
) -> None:
    """Persist a message to the agent_messages table."""
    import uuid as _uuid

    from server.core.db import async_session
    from server.models.agent import AgentMessage

    async with async_session() as db:
        db.add(
            AgentMessage(
                id=str(_uuid.uuid4()),
                user_id=user_id,
                role=role,
                content=content,
                agent_name=agent_name,
                thinking=thinking,
            )
        )
        await db.commit()


async def _dispatcher_sse_saving_agent_msgs(
    dispatcher,
    user_id: str,
    message: str,
    agent_name_req: Optional[str],
    chain_action: Optional[str] = None,
):
    """Wrap _dispatcher_sse: stream events + save assistant reply to agent_messages on finish."""
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    captured_agent: Optional[str] = agent_name_req

    try:
        async for frame in _dispatcher_sse(dispatcher, message, agent_name_req, chain_action):
            yield frame
            if not frame.startswith("data: "):
                continue
            raw = frame[6:].strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "event":
                continue
            role = payload.get("role", "")
            content = payload.get("content", "")
            if role == "assistant" and content:
                content_parts.append(content)
                agent_field = payload.get("agent_name") or payload.get("agent")
                if agent_field:
                    captured_agent = agent_field
            elif role == "thinking" and content:
                thinking_parts.append(content)
    finally:
        if content_parts:
            try:
                await _save_agent_message(
                    user_id=user_id,
                    role="assistant",
                    content="".join(content_parts),
                    agent_name=captured_agent,
                    thinking="".join(thinking_parts) or None,
                )
                sys_log(f"[Agent] Saved assistant message to agent_messages (user={user_id})")
            except Exception as e:
                sys_log(f"[Agent] Failed to save assistant message: {e}", level="WARNING")


async def _dispatcher_sse_saving(
    dispatcher,
    message: str,
    agent_name_req: Optional[str],
    thread_id: str,
    user_id: str,
    chain_action: Optional[str] = None,
):
    """Wrap _dispatcher_sse: pass all frames through, save assistant reply on finish."""
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_use_list: list[dict] = []
    captured_agent: Optional[str] = None

    try:
        async for frame in _dispatcher_sse(dispatcher, message, agent_name_req, chain_action):
            yield frame
            if not frame.startswith("data: "):
                continue
            raw = frame[6:].strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "event":
                continue
            role = payload.get("role", "")
            content = payload.get("content", "")
            if role == "assistant" and content:
                content_parts.append(content)
                if payload.get("agent"):
                    captured_agent = payload["agent"]
            elif role == "thinking" and content:
                thinking_parts.append(content)
            elif role == "tool_use":
                entry = {k: v for k, v in payload.items() if k not in ("type", "role")}
                if entry:
                    tool_use_list.append(entry)
    finally:
        if content_parts:
            await _save_chat_message(
                thread_id=thread_id,
                user_id=user_id,
                role="assistant",
                content="".join(content_parts),
                tool_use=tool_use_list or None,
                thinking="".join(thinking_parts) or None,
                agent_name=captured_agent,
            )


@router.post("/dispatch")
async def agent_dispatch(req: AgentDispatchRequest, request: Request):
    """SSE-stream a user message through the AgentDispatcher.

    This drives Guided (and Auto) pipeline orchestration:
    dispatch → agent runs → chain detection → guided gate or auto execute.
    """
    user_id = getattr(request.state, "user_id", "default")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    dispatcher = _get_dispatcher(user_id)
    sys_log(f"[Agent] Dispatch: agent={req.agent or 'auto-route'}, msg={req.message[:60]!r} (user={user_id})")

    if req.thread_id:
        await _save_chat_message(
            thread_id=req.thread_id,
            user_id=user_id,
            role="user",
            content=req.message,
        )

    stream = (
        _dispatcher_sse_saving(dispatcher, req.message, req.agent, req.thread_id, user_id)
        if req.thread_id
        else _dispatcher_sse_saving_agent_msgs(dispatcher, user_id, req.message, req.agent)
    )

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/pipeline")
async def agent_pipeline(request: Request):
    """Return dispatcher pipeline status (completed stages, pending chain, busy, session)."""
    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)
    try:
        status = dispatcher.get_pipeline_status()
        return status
    except Exception as e:
        sys_log(f"[Agent] Pipeline status error: {e}", level="ERROR")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chain/advance")
async def agent_chain_advance(req: ChainAdvanceRequest, request: Request):
    """Approve or skip the current pending chain (Guided mode).

    Internally sends the appropriate confirmation text through the dispatcher,
    so all existing pending_chain handling (_is_confirmation / _is_skip) applies.
    Streams the resulting agent activity as SSE.
    """
    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)

    # Reject when no pending chain so the caller can recover cleanly
    if not getattr(dispatcher, "_pending_chain", None):
        raise HTTPException(status_code=409, detail="No pending chain to advance")

    # Apply user-edited parameter overrides (from the confirmation modal)
    # onto the pending chain before executing it.
    if req.action == "approve" and req.overrides:
        try:
            dispatcher.apply_chain_overrides(req.overrides)
        except Exception as e:
            sys_log(f"[Agent] Failed to apply chain overrides: {e}", level="WARNING")

    _lang = resolve_lang(config=load_pipeline_config(user_id))
    message = advance_message(req.action, _lang)
    sys_log(f"[Agent] Chain advance: action={req.action} lang={_lang} (user={user_id})")

    if req.thread_id:
        await _save_chat_message(
            thread_id=req.thread_id,
            user_id=user_id,
            role="user",
            content=message,
        )

    stream = (
        _dispatcher_sse_saving(dispatcher, message, None, req.thread_id, user_id, req.action)
        if req.thread_id
        else _dispatcher_sse_saving_agent_msgs(dispatcher, user_id, message, None, req.action)
    )

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/execution/confirm")
async def agent_execution_confirm(req: ExecutionConfirmRequest, request: Request):
    """Confirm or cancel a guided execution (generation / training / TWIST).

    Valid only while a `start_*` tool is held by the guard in guided mode
    (`_pending_execution`). Same shape as `/eval/confirm`; the difference is the
    target tool and that edited values are merged through an allowlist.

    - approve: merge the edits into the pending tool_input, then send a
      confirmation message to trigger the existing resume path.
    - cancel: release the pending execution without running it.
    """
    from server.core.pending_execution import merge_overrides, selection_payload

    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)

    target_name = None
    target_agent = None
    pend = None
    for name, agent in (getattr(dispatcher, "_agents", {}) or {}).items():
        _p = getattr(agent, "_pending_execution", None)
        if selection_payload(name, _p):
            target_name, target_agent, pend = name, agent, _p
            break
    if not pend:
        raise HTTPException(status_code=409, detail="No pending execution")

    if req.action == "cancel":
        target_agent._pending_execution = None
        sys_log(f"[Agent] Execution cancelled: {pend.get('tool_name')} (user={user_id})")
        return {"status": "cancelled"}

    tool = str(pend.get("tool_name") or "")
    pend["tool_input"] = merge_overrides(tool, pend.get("tool_input"), req.overrides)
    sys_log(
        f"[Agent] Execution confirm: tool={tool}, agent={target_name}, params={pend['tool_input']} (user={user_id})"
    )

    message = advance_message("approve", resolve_lang(config=load_pipeline_config(user_id)))
    if req.thread_id:
        await _save_chat_message(thread_id=req.thread_id, user_id=user_id, role="user", content=message)
    stream = (
        _dispatcher_sse_saving(dispatcher, message, target_name, req.thread_id, user_id)
        if req.thread_id
        else _dispatcher_sse_saving_agent_msgs(dispatcher, user_id, message, target_name)
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/chain/pause")
async def agent_chain_pause(request: Request):
    """Pause auto-chaining — even in Auto mode, next gate will wait for user."""
    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)
    dispatcher.pause_chain()
    sys_log(f"[Agent] Chain paused (user={user_id})")
    return {"chain_paused": True}


@router.post("/chain/resume")
async def agent_chain_resume(request: Request):
    """Resume auto-chaining."""
    user_id = getattr(request.state, "user_id", "default")
    dispatcher = _get_dispatcher(user_id)
    dispatcher.resume_chain()
    sys_log(f"[Agent] Chain resumed (user={user_id})")
    return {"chain_paused": False}
