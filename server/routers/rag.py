"""RAG router — /api/rag

GET    /collections        — list ChromaDB collections
DELETE /collections/{name} — delete one
POST   /index/execute      — write chunks + embeddings (worker -> app)
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server.core.dataset_paths import chroma_dir
from server.core.logging import sys_log

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IndexExecuteRequest(BaseModel):
    """Internal request from Celery worker for ChromaDB write.
    Worker sends text chunks; app handles embedding + storage."""

    doc_id: str
    doc_name: str
    chunks: List[str]
    collection_name: str = "koni_documents"
    embedding_model: Optional[str] = None
    metadata: Optional[dict] = None
    page_nums: Optional[List[Optional[int]]] = None


# ---------------------------------------------------------------------------
# POST /index
# ---------------------------------------------------------------------------


@router.delete("/collections/{name}")
async def delete_collection(name: str, request: Request):
    """Delete a ChromaDB collection."""
    user_id = getattr(request.state, "user_id", "default")
    try:
        from modules.rag.vectorstore import VectorStore

        vs = VectorStore()
        deleted_count = vs.delete_dataset(name)
        sys_log(f"[RAG] Collection '{name}' deleted ({deleted_count} docs, user={user_id})")
        return {"message": f"Collection '{name}' deleted", "deleted_documents": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------


@router.get("/collections")
async def list_collections(request: Request):
    """List ChromaDB collections."""
    try:
        import chromadb

        chroma_path = str(chroma_dir())
        client = chromadb.PersistentClient(path=chroma_path)
        collections = []
        for col in client.list_collections():
            collections.append(
                {
                    "name": col.name,
                    "count": col.count(),
                }
            )
        return {"collections": collections, "total": len(collections)}
    except Exception as e:
        return {"collections": [], "total": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GET /datasets  — indexed datasets for RAGAS selector
# ---------------------------------------------------------------------------


@router.post("/index/execute")
async def index_execute(req: IndexExecuteRequest, request: Request):
    """Write chunks to ChromaDB on behalf of the worker.

    ChromaDB is opened by the app process, so the worker posts here instead of
    writing the store itself. Not called by the frontend.
    """
    try:
        from modules.rag.embedding_cache import get_embedding_manager
        from modules.rag.vectorstore import VectorStore

        em = get_embedding_manager(model_name=req.embedding_model)
        embeddings = em.embed_documents(req.chunks)

        vs = VectorStore(collection_name=req.collection_name)
        chunks_added = vs.add_document(
            doc_id=req.doc_id,
            doc_name=req.doc_name,
            chunks=req.chunks,
            embeddings=embeddings,
            metadata=req.metadata,
            page_nums=req.page_nums,
        )

        sys_log(f"[RAG] Indexed {chunks_added} chunks for doc={req.doc_id} into collection={req.collection_name}")
        return {"status": "ok", "chunks_added": chunks_added, "collection": req.collection_name}
    except Exception as e:
        sys_log(f"[RAG] Index execute failed: {e}", level="ERROR")
        raise HTTPException(status_code=500, detail=f"Index execute failed: {e}")
