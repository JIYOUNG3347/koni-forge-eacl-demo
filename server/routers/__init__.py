"""HTTP routers. `ALL_ROUTERS` is what `api.py` mounts."""

from .agent import router as agent_router
from .agent_chat import router as agent_chat_router
from .auth import router as auth_router
from .chat import router as chat_router
from .data import router as data_router
from .hf import router as hf_router
from .models import router as models_router
from .rag import router as rag_router
from .system import router as system_router
from .train import router as train_router

ALL_ROUTERS = [
    auth_router,
    data_router,
    train_router,
    rag_router,
    chat_router,
    agent_router,
    agent_chat_router,
    system_router,
    models_router,
    hf_router,
]

__all__ = ["ALL_ROUTERS"]
