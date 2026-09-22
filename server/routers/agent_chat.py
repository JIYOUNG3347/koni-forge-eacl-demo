"""One conversation per user; no threads.
Completely separate from /api/chat/threads/* used by the chat page.

GET    /api/agent/messages   — this user's agent conversation, oldest first
POST   /api/agent/messages   — append a message
DELETE /api/agent/messages   — clear the conversation (the eraser button)
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.db import get_session
from server.core.logging import sys_log
from server.models.agent import AgentMessage

router = APIRouter(prefix="/api/agent", tags=["agent_chat"])


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────


class AgentMessageCreate(BaseModel):
    role: str
    content: str
    agent_name: Optional[str] = None
    tool_use: Optional[str] = None
    thinking: Optional[str] = None


class AgentMessageOut(BaseModel):
    id: str
    role: str
    content: str
    agent_name: Optional[str] = None
    tool_use: Optional[str] = None
    thinking: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────


@router.get("/messages", response_model=list[AgentMessageOut])
async def list_agent_messages(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """This user's agent conversation, oldest first."""
    user_id = getattr(request.state, "user_id", "default")

    result = await session.execute(
        select(AgentMessage).where(AgentMessage.user_id == user_id).order_by(AgentMessage.created_at.asc())
    )
    messages = result.scalars().all()

    return [
        AgentMessageOut(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            agent_name=msg.agent_name,
            tool_use=msg.tool_use,
            thinking=msg.thinking,
            created_at=msg.created_at.isoformat(),
        )
        for msg in messages
    ]


@router.post("/messages", response_model=AgentMessageOut, status_code=201)
async def create_agent_message(
    payload: AgentMessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Append a message to the agent conversation."""
    user_id = getattr(request.state, "user_id", "default")

    msg = AgentMessage(
        id=str(uuid.uuid4()),
        user_id=user_id,
        role=payload.role,
        content=payload.content,
        agent_name=payload.agent_name,
        tool_use=payload.tool_use,
        thinking=payload.thinking,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)

    return AgentMessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        agent_name=msg.agent_name,
        tool_use=msg.tool_use,
        thinking=msg.thinking,
        created_at=msg.created_at.isoformat(),
    )


@router.delete("/messages", status_code=204)
async def clear_agent_messages(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Clear the agent conversation (the eraser button).

    The chat page's threads and messages are unaffected.
    """
    user_id = getattr(request.state, "user_id", "default")

    await session.execute(sql_delete(AgentMessage).where(AgentMessage.user_id == user_id))
    await session.commit()
    sys_log(f"[agent_chat] cleared messages: user={user_id}")
