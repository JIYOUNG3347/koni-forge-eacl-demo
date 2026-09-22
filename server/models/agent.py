"""Completely separate from the Thread/Message models used by the chat page.
One conversation per user; no threads.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from server.core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # user|assistant|tool
    content: Mapped[str] = mapped_column(Text, nullable=False)

    agent_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_use: Mapped[str | None] = mapped_column(Text, nullable=True)
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    __table_args__ = (
        Index("ix_agent_messages_user_created", "user_id", "created_at"),
    )
