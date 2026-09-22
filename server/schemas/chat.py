"""Chat API Pydantic schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ThreadCreate(BaseModel):
    id: str | None = None  # client-generated UUID; server assigns one if absent
    title: str = Field(..., min_length=1, max_length=200)
    mode: Literal["auto", "guided"]


class ThreadUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)


class ThreadOut(BaseModel):
    id: str
    user_id: str
    title: str
    mode: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system", "tool_result"]
    content: str
    tool_use: list[dict[str, Any]] | None = None
    thinking: str | None = None
    agent_name: str | None = None


class MessageOut(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    tool_use: list[dict[str, Any]] | None = None
    thinking: str | None = None
    agent_name: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
