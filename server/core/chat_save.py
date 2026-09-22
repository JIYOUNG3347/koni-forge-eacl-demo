"""Shared utility for persisting chat messages to SQLite."""

import uuid
from datetime import datetime, timezone

from server.core.logging import sys_log


async def save_chat_message(
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    *,
    tool_use: list | None = None,
    thinking: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Persist a chat message to SQLite. No-ops silently on any error."""
    if not content:
        return
    try:
        from sqlalchemy import select

        from server.core.db import async_session
        from server.models.chat import Message, Thread

        async with async_session() as session:
            result = await session.execute(select(Thread).where(Thread.id == thread_id))
            thread = result.scalar_one_or_none()
            if thread is None or thread.user_id != user_id:
                return
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(
                Message(
                    id=str(uuid.uuid4()),
                    thread_id=thread_id,
                    role=role,
                    content=content,
                    tool_use=tool_use,
                    thinking=thinking,
                    agent_name=agent_name,
                    created_at=now,
                )
            )
            thread.updated_at = now
            await session.commit()
    except Exception as exc:
        sys_log(f"[Chat] Save error (thread={thread_id}): {exc}", level="WARNING")
