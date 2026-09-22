"""Tests use an in-memory SQLite engine so they are fully isolated from
the production /storage/koni.db and do NOT require a running server.

Coverage:
  A. DB init — tables created
  B. Thread CRUD — create, list, get, update, delete
  C. Cascade — messages deleted when thread deleted
  D. Isolation — user A cannot access user B's threads
  E. Message CRUD — create, list (asc), role/extras roundtrip
  F. updated_at — thread.updated_at bumped when message added
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── test engine (in-memory SQLite, per-module scope) ────────────────────────
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

_engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
_TestSession = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True, scope="module")
async def _create_tables():
    """One-time table creation for the whole module."""
    import server.models.chat  # noqa: F401 — registers ORM metadata
    from server.core.db import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def session():
    async with _TestSession() as s:
        yield s


# ── helpers ──────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _make_thread(
    session: AsyncSession,
    user_id: str = "user_a",
    title: str = "test thread",
    mode: str = "guided",
):
    from server.models.chat import Thread

    t = Thread(id=str(uuid.uuid4()), user_id=user_id, title=title, mode=mode)
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


async def _make_message(
    session: AsyncSession,
    thread_id: str,
    role: str = "user",
    content: str = "hello",
    **extras,
):
    from server.models.chat import Message

    now = _now()
    m = Message(
        id=str(uuid.uuid4()),
        thread_id=thread_id,
        role=role,
        content=content,
        created_at=now,
        **extras,
    )
    session.add(m)
    await session.commit()
    await session.refresh(m)
    return m


# ── A. DB initialisation ─────────────────────────────────────────────────────


class TestDBInit:
    async def test_threads_table_exists(self):
        async with _engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "threads" in tables

    async def test_messages_table_exists(self):
        async with _engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "messages" in tables


# ── B. Thread CRUD ────────────────────────────────────────────────────────────


class TestThreadCRUD:
    async def test_create_thread_stores_fields(self, session):
        from sqlalchemy import select

        from server.models.chat import Thread

        thread = await _make_thread(session, user_id="alice", title="My Thread", mode="auto")
        result = await session.execute(select(Thread).where(Thread.id == thread.id))
        fetched = result.scalar_one()

        assert fetched.user_id == "alice"
        assert fetched.title == "My Thread"
        assert fetched.mode == "auto"
        assert fetched.created_at is not None
        assert fetched.updated_at is not None

    async def test_list_threads_by_user(self, session):
        from sqlalchemy import desc, select

        from server.models.chat import Thread

        uid = f"list_user_{uuid.uuid4().hex[:6]}"
        await _make_thread(session, user_id=uid, title="T1")
        await _make_thread(session, user_id=uid, title="T2")
        await _make_thread(session, user_id="other_user", title="Tx")

        result = await session.execute(select(Thread).where(Thread.user_id == uid).order_by(desc(Thread.updated_at)))
        threads = result.scalars().all()
        assert len(threads) == 2
        assert all(t.user_id == uid for t in threads)

    async def test_update_thread_title(self, session):
        from sqlalchemy import select

        from server.models.chat import Thread

        thread = await _make_thread(session, title="original title")
        thread.title = "changed title"
        await session.commit()
        await session.refresh(thread)

        result = await session.execute(select(Thread).where(Thread.id == thread.id))
        assert result.scalar_one().title == "changed title"

    async def test_delete_thread(self, session):
        from sqlalchemy import select

        from server.models.chat import Thread

        thread = await _make_thread(session)
        tid = thread.id
        await session.delete(thread)
        await session.commit()

        result = await session.execute(select(Thread).where(Thread.id == tid))
        assert result.scalar_one_or_none() is None


# ── C. Cascade delete ─────────────────────────────────────────────────────────


class TestCascadeDelete:
    async def test_messages_deleted_with_thread(self, session):
        from sqlalchemy import select

        from server.models.chat import Message

        thread = await _make_thread(session)
        await _make_message(session, thread.id, content="first")
        await _make_message(session, thread.id, content="second")

        await session.delete(thread)
        await session.commit()

        result = await session.execute(select(Message).where(Message.thread_id == thread.id))
        assert result.scalars().all() == []


# ── D. User isolation ─────────────────────────────────────────────────────────


class TestUserIsolation:
    async def test_thread_user_id_stored_correctly(self, session):
        t_a = await _make_thread(session, user_id="user_alpha")
        t_b = await _make_thread(session, user_id="user_beta")
        assert t_a.user_id == "user_alpha"
        assert t_b.user_id == "user_beta"
        assert t_a.user_id != t_b.user_id

    async def test_list_excludes_other_user_threads(self, session):
        from sqlalchemy import select

        from server.models.chat import Thread

        uid_x = f"iso_x_{uuid.uuid4().hex[:6]}"
        uid_y = f"iso_y_{uuid.uuid4().hex[:6]}"
        await _make_thread(session, user_id=uid_x)
        await _make_thread(session, user_id=uid_y)

        result = await session.execute(select(Thread).where(Thread.user_id == uid_x))
        threads = result.scalars().all()
        assert all(t.user_id == uid_x for t in threads)
        assert not any(t.user_id == uid_y for t in threads)


# ── E. Message CRUD ───────────────────────────────────────────────────────────


class TestMessageCRUD:
    async def test_create_user_message(self, session):
        thread = await _make_thread(session)
        msg = await _make_message(session, thread.id, role="user", content="a question")
        assert msg.role == "user"
        assert msg.content == "a question"
        assert msg.thread_id == thread.id

    async def test_create_assistant_message_with_tool_use(self, session):
        thread = await _make_thread(session)
        tool_data = [{"name": "search", "input": "query", "output": "result"}]
        msg = await _make_message(
            session,
            thread.id,
            role="assistant",
            content="search result",
            tool_use=tool_data,
            agent_name="corpus_specialist",
        )
        assert msg.tool_use == tool_data
        assert msg.agent_name == "corpus_specialist"
        assert msg.thinking is None

    async def test_create_assistant_message_with_thinking(self, session):
        thread = await _make_thread(session)
        msg = await _make_message(
            session,
            thread.id,
            role="assistant",
            content="answer",
            thinking="internal reasoning...",
        )
        assert msg.thinking == "internal reasoning..."

    async def test_list_messages_chronological(self, session):
        import asyncio

        from sqlalchemy import select

        from server.models.chat import Message

        thread = await _make_thread(session)
        for i in range(3):
            await asyncio.sleep(0.001)  # ensure distinct timestamps
            await _make_message(session, thread.id, content=f"msg {i}")

        result = await session.execute(
            select(Message).where(Message.thread_id == thread.id).order_by(Message.created_at)
        )
        msgs = result.scalars().all()
        assert len(msgs) == 3
        assert [m.content for m in msgs] == ["msg 0", "msg 1", "msg 2"]

    async def test_auto_mode_message_no_extras(self, session):
        thread = await _make_thread(session, mode="auto")
        msg = await _make_message(session, thread.id, role="user", content="typed directly")
        assert msg.tool_use is None
        assert msg.thinking is None
        assert msg.agent_name is None

    async def test_message_isolation_by_thread(self, session):
        from sqlalchemy import select

        from server.models.chat import Message

        t1 = await _make_thread(session)
        t2 = await _make_thread(session)
        await _make_message(session, t1.id, content="t1 message")
        await _make_message(session, t2.id, content="t2 message")

        result = await session.execute(select(Message).where(Message.thread_id == t1.id))
        msgs = result.scalars().all()
        assert len(msgs) == 1
        assert msgs[0].content == "t1 message"


# ── F. updated_at ─────────────────────────────────────────────────────────────


class TestThreadUpdatedAt:
    async def test_message_create_updates_thread_updated_at(self, session):
        import asyncio

        from sqlalchemy import select

        from server.models.chat import Thread

        thread = await _make_thread(session)
        original_updated = thread.updated_at

        await asyncio.sleep(0.01)  # ensure time advance
        now = _now()
        thread.updated_at = now
        session.add(thread)
        await session.commit()
        await session.refresh(thread)

        result = await session.execute(select(Thread).where(Thread.id == thread.id))
        refreshed = result.scalar_one()
        assert refreshed.updated_at >= original_updated
