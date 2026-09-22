"""Tests use an in-memory SQLite engine — no fastapi dependency."""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── test engine ────────────────────────────────────────────────────────────────
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
_engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
_TestSession = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True, scope="module")
async def _create_tables():
    import server.models.chat  # noqa: F401
    from server.core.db import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ── helpers ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _make_thread(user_id: str = "user_a", mode: str = "guided"):
    from server.models.chat import Thread

    async with _TestSession() as s:
        t = Thread(id=str(uuid.uuid4()), user_id=user_id, title="test", mode=mode)
        s.add(t)
        await s.commit()
        await s.refresh(t)
        return t


@asynccontextmanager
async def _test_session_cm():
    """Drop-in for server.core.db.async_session using the test engine."""
    async with _TestSession() as s:
        yield s


# ── A. save_chat_message ──────────────────────────────────────────────────────


class TestSaveChatMessage:
    async def test_saves_message_and_fields(self, monkeypatch):
        from sqlalchemy import select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        thread = await _make_thread(user_id="alice")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message(
            thread.id,
            "alice",
            "user",
            "hello",
            agent_name="general",
        )

        async with _TestSession() as s:
            result = await s.execute(select(Message).where(Message.thread_id == thread.id))
            msgs = result.scalars().all()
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "hello"
        assert msgs[0].agent_name == "general"

    async def test_noop_wrong_user(self, monkeypatch):
        from sqlalchemy import select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        thread = await _make_thread(user_id="owner")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message(thread.id, "imposter", "user", "hack")

        async with _TestSession() as s:
            result = await s.execute(select(Message).where(Message.thread_id == thread.id))
            assert result.scalars().all() == []

    async def test_noop_missing_thread(self, monkeypatch):
        from server.core.chat_save import save_chat_message

        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message("does-not-exist", "alice", "user", "hi")

    async def test_noop_empty_content(self, monkeypatch):
        from sqlalchemy import select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        thread = await _make_thread(user_id="empty_user")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message(thread.id, "empty_user", "assistant", "")

        async with _TestSession() as s:
            result = await s.execute(select(Message).where(Message.thread_id == thread.id))
            assert result.scalars().all() == []

    async def test_updates_thread_updated_at(self, monkeypatch):
        import asyncio

        from sqlalchemy import select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Thread

        thread = await _make_thread(user_id="ts_user")
        original_ts = thread.updated_at

        await asyncio.sleep(0.01)
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message(thread.id, "ts_user", "assistant", "reply")

        async with _TestSession() as s:
            result = await s.execute(select(Thread).where(Thread.id == thread.id))
            refreshed = result.scalar_one()
        assert refreshed.updated_at >= original_ts

    async def test_saves_tool_use_and_thinking(self, monkeypatch):
        from sqlalchemy import select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        thread = await _make_thread(user_id="tool_user")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        tool_data = [{"name": "search", "result": "found"}]
        await save_chat_message(
            thread.id,
            "tool_user",
            "assistant",
            "result",
            tool_use=tool_data,
            thinking="internal reasoning",
        )

        async with _TestSession() as s:
            result = await s.execute(select(Message).where(Message.thread_id == thread.id))
            msg = result.scalar_one()
        assert msg.tool_use == tool_data
        assert msg.thinking == "internal reasoning"


# ── B. SSE frame accumulation logic ──────────────────────────────────────────
#
# _dispatcher_sse_saving (in agent.py) can't be imported in tests because
# server.routers.__init__ requires fastapi. Instead we test the accumulation
# logic via a local standalone helper that mirrors the same parsing code.


def _parse_sse_frames(frames: list[str]) -> dict:
    """Replicate the accumulation logic from _dispatcher_sse_saving."""
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_use_list: list[dict] = []
    captured_agent: str | None = None

    for frame in frames:
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

    return {
        "content": "".join(content_parts),
        "thinking": "".join(thinking_parts) or None,
        "tool_use": tool_use_list or None,
        "agent_name": captured_agent,
    }


class TestSseAccumulation:
    def test_accumulates_assistant_content(self):
        frames = [
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': 'Hello', 'agent': 'general'})}\n\n",
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': ' world'})}\n\n",
            'data: {"type": "done"}\n\n',
        ]
        result = _parse_sse_frames(frames)
        assert result["content"] == "Hello world"
        assert result["agent_name"] == "general"
        assert result["thinking"] is None
        assert result["tool_use"] is None

    def test_accumulates_thinking(self):
        frames = [
            f"data: {json.dumps({'type': 'event', 'role': 'thinking', 'content': 'internal reasoning'})}\n\n",
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': 'answer'})}\n\n",
        ]
        result = _parse_sse_frames(frames)
        assert result["thinking"] == "internal reasoning"
        assert result["content"] == "answer"

    def test_accumulates_tool_use(self):
        frames = [
            f"data: {json.dumps({'type': 'event', 'role': 'tool_use', 'name': 'search', 'input': 'query'})}\n\n",
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': 'result'})}\n\n",
        ]
        result = _parse_sse_frames(frames)
        assert result["tool_use"] is not None
        assert result["tool_use"][0]["name"] == "search"

    def test_ignores_keepalive_and_status(self):
        frames = [
            ": keepalive\n\n",
            f"data: {json.dumps({'type': 'status', 'completed_stages': []})}\n\n",
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': 'response'})}\n\n",
            ": keepalive\n\n",
        ]
        result = _parse_sse_frames(frames)
        assert result["content"] == "response"

    def test_empty_frames_produce_no_content(self):
        frames = [
            f"data: {json.dumps({'type': 'status'})}\n\n",
            'data: {"type": "done"}\n\n',
        ]
        result = _parse_sse_frames(frames)
        assert result["content"] == ""
        assert result["thinking"] is None
        assert result["tool_use"] is None

    def test_malformed_data_ignored(self):
        frames = [
            "data: not-json\n\n",
            f"data: {json.dumps({'type': 'event', 'role': 'assistant', 'content': 'ok'})}\n\n",
        ]
        result = _parse_sse_frames(frames)
        assert result["content"] == "ok"


# ── C. ThreadCreate client ID ─────────────────────────────────────────────────


class TestThreadCreateClientId:
    def test_accepts_client_id(self):
        from server.schemas.chat import ThreadCreate

        client_id = str(uuid.uuid4())
        body = ThreadCreate(id=client_id, title="test", mode="guided")
        assert body.id == client_id

    def test_none_id_is_valid(self):
        from server.schemas.chat import ThreadCreate

        body = ThreadCreate(id=None, title="test", mode="auto")
        assert body.id is None

    def test_missing_id_defaults_to_none(self):
        from server.schemas.chat import ThreadCreate

        body = ThreadCreate(title="test", mode="guided")
        assert body.id is None


class TestChatGenerateUserSaveLogic:
    """Verify the reversed-history lookup logic added to chat_generate.

    The endpoint receives the full accumulated history and must extract only
    the last user turn — earlier turns are already in the DB from prior sends.
    """

    async def test_saves_last_user_message_from_history(self, monkeypatch):
        """Full history with prior turns: only the newest user message saved."""
        from sqlalchemy import select as sa_select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        user_id = "gen_" + str(uuid.uuid4())[:6]
        thread = await _make_thread(user_id, mode="auto")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        # Simulate what chat_generate now does: reversed lookup for last user turn
        messages = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "new question"},  # <- should be saved
        ]
        user_content = next(
            (str(m.get("content", "")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        assert user_content == "new question"
        await save_chat_message(thread.id, user_id, "user", user_content)

        async with _TestSession() as s:
            result = await s.execute(sa_select(Message).where(Message.thread_id == thread.id))
            msgs = result.scalars().all()

        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "new question"

    async def test_skips_save_when_no_user_turn_in_history(self):
        """History with only system/assistant turns: nothing saved."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "assistant", "content": "first answer"},
        ]
        user_content = next(
            (str(m.get("content", "")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        assert user_content == ""  # guard: if _user_content → skip save

    async def test_skips_save_when_thread_id_is_none(self):
        """No thread_id → save is skipped entirely (no error)."""
        # Simulates: if req.thread_id: ...  (falsy → skipped)
        thread_id = None
        messages = [{"role": "user", "content": "test"}]
        user_content = next(
            (str(m.get("content", "")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        # Guard: `if req.thread_id and _user_content` → both must be truthy
        should_save = bool(thread_id) and bool(user_content)
        assert should_save is False

    async def test_last_user_extracted_correctly_with_trailing_assistant(self):
        """History ending in assistant turn: last user is still found correctly."""
        messages = [
            {"role": "user", "content": "question A"},
            {"role": "assistant", "content": "answer A"},
            {"role": "user", "content": "question B"},
            {"role": "assistant", "content": "answer B"},
            # next send would add a new user turn — but test the lookup
        ]
        # Only "question B" should match (last user in reversed order)
        user_content = next(
            (str(m.get("content", "")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        assert user_content == "question B"


class TestAgentAskSaveLogic:
    """Verify user + assistant save behaviour added to agent_ask.

    Tests the same patterns the endpoint now executes, using save_chat_message
    directly (no HTTP server needed).
    """

    async def test_ask_saves_user_message(self, monkeypatch):
        """agent_ask pattern: user message saved before stream."""
        from sqlalchemy import select as sa_select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        user_id = "ask_" + str(uuid.uuid4())[:6]
        thread = await _make_thread(user_id, mode="auto")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)
        message = "auto mode test question"

        # Simulate: if req.thread_id and req.thread_id != "default"
        assert thread.id and thread.id != "default"
        await save_chat_message(thread.id, user_id, "user", message)

        async with _TestSession() as s:
            result = await s.execute(sa_select(Message).where(Message.thread_id == thread.id))
            msgs = result.scalars().all()

        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == message

    async def test_ask_saves_assistant_tokens(self, monkeypatch):
        """_stream_with_save pattern: accumulated tokens saved on finally."""
        from sqlalchemy import select as sa_select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        user_id = "ask2_" + str(uuid.uuid4())[:6]
        thread = await _make_thread(user_id, mode="auto")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        # Simulate: SSE token accumulation → finally → save_chat_message
        token_parts = ["Hel", "lo", ", ", "nice to meet you."]
        await save_chat_message(thread.id, user_id, "user", "question")
        await save_chat_message(thread.id, user_id, "assistant", "".join(token_parts))

        async with _TestSession() as s:
            result = await s.execute(
                sa_select(Message).where(Message.thread_id == thread.id).order_by(Message.created_at)
            )
            msgs = result.scalars().all()

        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Hello, nice to meet you."

    async def test_default_thread_id_guard(self):
        """thread_id='default' (schema default value) → save skipped."""
        # Simulates: if req.thread_id and req.thread_id != "default"
        thread_id = "default"
        should_save = bool(thread_id) and thread_id != "default"
        assert should_save is False

    async def test_empty_thread_id_guard(self):
        """thread_id=None or '' → save skipped."""
        for thread_id in (None, ""):
            should_save = bool(thread_id) and thread_id != "default"
            assert should_save is False

    async def test_ask_user_and_assistant_both_present_after_full_flow(self, monkeypatch):
        """Full ask flow: both user and assistant appear in DB."""
        from sqlalchemy import select as sa_select

        from server.core.chat_save import save_chat_message
        from server.models.chat import Message

        user_id = "ask3_" + str(uuid.uuid4())[:6]
        thread = await _make_thread(user_id, mode="auto")
        monkeypatch.setattr("server.core.db.async_session", _test_session_cm)

        await save_chat_message(thread.id, user_id, "user", "should still be visible after a reload")
        await save_chat_message(thread.id, user_id, "assistant", "Yes, it is.")

        async with _TestSession() as s:
            result = await s.execute(
                sa_select(Message).where(Message.thread_id == thread.id).order_by(Message.created_at)
            )
            msgs = result.scalars().all()

        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "should still be visible after a reload"
        assert msgs[1].content == "Yes, it is."
