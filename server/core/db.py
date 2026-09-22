"""Async SQLAlchemy engine backed by aiosqlite.

The database lives at ``<storage root>/koni.db``.
"""

from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from server.core.config import STORAGE_ROOT


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record):
    """SQLite requires FK enforcement per connection — not a one-time setting."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


DB_PATH = STORAGE_ROOT / "koni.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup (idempotent)."""
    from server.models import agent as _agent_models  # noqa: F401
    from server.models import chat as _chat_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Depends — yields an AsyncSession per request."""
    async with async_session() as session:
        yield session
