"""Chat router — /api/chat

POST /generate                            — SSE streaming chat (HF model direct inference)
GET  /threads                             — list user's threads
POST /threads                             — create thread
GET  /threads/{id}                        — get thread
PATCH /threads/{id}                       — update thread title
DELETE /threads/{id}                      — delete thread (cascade messages)
GET  /threads/{id}/messages               — list messages (asc)
POST /threads/{id}/messages               — append message
DELETE /threads/{id}/messages             — delete all messages (thread kept)
DELETE /threads/{id}/messages/{msg_id}    — delete single message
"""

import asyncio
import json
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete as sql_delete
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.db import get_session
from server.core.logging import sys_log
from server.models.chat import Message, Thread
from server.schemas.chat import (
    MessageCreate,
    MessageOut,
    ThreadCreate,
    ThreadOut,
    ThreadUpdate,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────


def _require_user(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


async def _get_thread_or_404(session: AsyncSession, thread_id: str, user_id: str) -> Thread:
    result = await session.execute(select(Thread).where(Thread.id == thread_id))
    thread = result.scalar_one_or_none()
    if thread is None or thread.user_id != user_id:
        # Conceal existence from other users
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


# ─────────────────────────────────────────────────────────────
# Thread CRUD
# ─────────────────────────────────────────────────────────────


@router.get("/threads", response_model=list[ThreadOut])
async def list_threads(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """The user's threads, most recently updated first."""
    user_id = _require_user(request)
    result = await session.execute(
        select(Thread).where(Thread.user_id == user_id).order_by(desc(Thread.updated_at)).limit(limit).offset(offset)
    )
    return result.scalars().all()


@router.post("/threads", response_model=ThreadOut, status_code=201)
async def create_thread(
    body: ThreadCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a thread."""
    user_id = _require_user(request)
    thread = Thread(
        id=body.id or str(uuid.uuid4()),
        user_id=user_id,
        title=body.title,
        mode=body.mode,
    )
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    sys_log(f"[Chat] Thread created: {thread.id} (user={user_id})")
    return thread


@router.get("/threads/{thread_id}", response_model=ThreadOut)
async def get_thread(
    thread_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    user_id = _require_user(request)
    return await _get_thread_or_404(session, thread_id, user_id)


@router.patch("/threads/{thread_id}", response_model=ThreadOut)
async def update_thread(
    thread_id: str,
    body: ThreadUpdate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Rename a thread."""
    user_id = _require_user(request)
    thread = await _get_thread_or_404(session, thread_id, user_id)
    if body.title is not None:
        thread.title = body.title
    await session.commit()
    await session.refresh(thread)
    return thread


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a thread and cascade to its messages."""
    user_id = _require_user(request)
    thread = await _get_thread_or_404(session, thread_id, user_id)
    # Explicit bulk DELETE works on both new DBs (with FK cascade) and legacy DBs
    # (without ON DELETE CASCADE), providing a dual safety net.
    await session.execute(sql_delete(Message).where(Message.thread_id == thread_id))
    await session.delete(thread)
    await session.commit()
    sys_log(f"[Chat] Thread deleted: {thread_id} (user={user_id})")


# ─────────────────────────────────────────────────────────────
# Message CRUD
# ─────────────────────────────────────────────────────────────


@router.get("/threads/{thread_id}/messages", response_model=list[MessageOut])
async def list_messages(
    thread_id: str,
    request: Request,
    limit: int = 200,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """Messages in a thread, oldest first."""
    user_id = _require_user(request)
    await _get_thread_or_404(session, thread_id, user_id)
    result = await session.execute(
        select(Message).where(Message.thread_id == thread_id).order_by(Message.created_at).limit(limit).offset(offset)
    )
    return result.scalars().all()


@router.delete("/threads/{thread_id}/messages", status_code=204)
async def delete_all_messages(
    thread_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete every message in a thread, keeping the thread."""
    user_id = _require_user(request)
    await _get_thread_or_404(session, thread_id, user_id)
    await session.execute(sql_delete(Message).where(Message.thread_id == thread_id))
    await session.commit()
    sys_log(f"[Chat] All messages cleared: thread={thread_id} (user={user_id})")


@router.delete("/threads/{thread_id}/messages/{message_id}", status_code=204)
async def delete_message(
    thread_id: str,
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete one message, after checking thread ownership."""
    user_id = _require_user(request)
    await _get_thread_or_404(session, thread_id, user_id)

    result = await session.execute(
        select(Message).where(
            Message.id == message_id,
            Message.thread_id == thread_id,
        )
    )
    message = result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")

    await session.delete(message)
    await session.commit()


@router.post(
    "/threads/{thread_id}/messages",
    response_model=MessageOut,
    status_code=201,
)
async def create_message(
    thread_id: str,
    body: MessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Append a message. Called by the backend after the SSE stream ends."""
    user_id = _require_user(request)
    thread = await _get_thread_or_404(session, thread_id, user_id)

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    message = Message(
        id=str(uuid.uuid4()),
        thread_id=thread_id,
        role=body.role,
        content=body.content,
        tool_use=body.tool_use,
        thinking=body.thinking,
        agent_name=body.agent_name,
        created_at=now,
    )
    session.add(message)
    thread.updated_at = now
    await session.commit()
    await session.refresh(message)
    return message


# ─────────────────────────────────────────────────────────────
# Direct inference (Auto mode)
# ─────────────────────────────────────────────────────────────


class ChatGenerateRequest(BaseModel):
    messages: List[Dict]
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = ""
    rag_enabled: bool = False
    rag_collection: str = ""
    thread_id: str | None = None


@router.post("/generate")
async def chat_generate(req: ChatGenerateRequest, request: Request):
    """SSE streaming chat, running inference on the loaded HuggingFace model.

    The model must be loaded first through /api/hf/load.
    """
    from pipelines.dialog import inference_state

    if not inference_state.get("model"):
        raise HTTPException(
            status_code=409,
            detail="No model is loaded. Pick a model in the left panel and press Load.",
        )

    user_id = getattr(request.state, "user_id", "default")

    messages = list(req.messages)
    if req.system_prompt:
        messages = [{"role": "system", "content": req.system_prompt}] + messages
    rag_datasets = [req.rag_collection] if req.rag_enabled and req.rag_collection else None

    sys_log(
        f"[Chat] generate: rag={req.rag_enabled}, collection={req.rag_collection!r}, "
        f" msgs={len(messages)}, user={user_id}"
    )

    async def _stream():
        from pipelines.dialog import stream_chat_response

        loop = asyncio.get_running_loop()
        gen = stream_chat_response(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            rag_enabled=req.rag_enabled,
            rag_datasets=rag_datasets,
        )

        _sentinel = object()
        try:
            while True:
                chunk = await loop.run_in_executor(None, next, gen, _sentinel)
                if chunk is _sentinel:
                    break
                yield chunk
        except Exception as e:
            sys_log(f"[Chat] Stream error: {e}", level="ERROR")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_with_save():
        """Pass-through wrapper that saves the assistant reply on finish."""
        token_parts: list[str] = []
        try:
            async for chunk in _stream():
                yield chunk
                if chunk.startswith("data: ") and "token" in chunk:
                    raw = chunk[6:].strip()
                    try:
                        data = json.loads(raw)
                        token = data.get("token", "")
                        if token:
                            token_parts.append(token)
                    except json.JSONDecodeError:
                        pass
        finally:
            if req.thread_id and token_parts:
                from server.core.chat_save import save_chat_message

                await save_chat_message(req.thread_id, user_id, "assistant", "".join(token_parts))

    # Save user message before streaming (mirrors agent_dispatch pattern).
    # req.messages is the full accumulated history — extract only the last user turn.
    if req.thread_id:
        _user_content = next(
            (
                str(m.get("content", ""))
                for m in reversed(req.messages)
                if isinstance(m, dict) and m.get("role") == "user"
            ),
            "",
        )
        if _user_content:
            from server.core.chat_save import save_chat_message

            await save_chat_message(req.thread_id, user_id, "user", _user_content)

    return StreamingResponse(
        _stream_with_save() if req.thread_id else _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
