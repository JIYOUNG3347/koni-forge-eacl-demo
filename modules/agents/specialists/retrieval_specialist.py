"""
KONI-Forge RetrievalSpecialist — Retrieval pipeline management agent

Roles:
- Document indexing strategy recommendations
- Search quality analysis
- Embedding model selection advice
- Self-RAG configuration guide
- RAGAS evaluation result interpretation
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from server.core.agent_tool_errors import error_text as _err
from server.core.dataset_paths import chroma_dir, raw_root
from server.core.pipeline_stage_texts import stage_text as _stage_text

from ..foundation import AgentBase

INTERNAL_API_URL = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
CHROMA_PERSIST_DIR = str(chroma_dir())
RAG_JOB_KEY_PREFIX = "rag_idx:"


def _open_chroma_client():
    """Return a ChromaDB PersistentClient rooted at CHROMA_PERSIST_DIR, or None."""
    try:
        if not Path(CHROMA_PERSIST_DIR).exists():
            return None
        import chromadb

        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception:
        return None


def _list_chroma_collections() -> List[Dict[str, Any]]:
    """Enumerate collections with chunk counts. Empty collections are included
    so the caller can distinguish "nothing indexed yet" from a failure."""
    client = _open_chroma_client()
    if client is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        for col in client.list_collections():
            try:
                count = col.count()
            except Exception:
                count = 0
            out.append({"name": col.name, "chunk_count": count})
    except Exception:
        pass
    return out


class RetrievalSpecialist(AgentBase):
    default_autonomy = "guided"
    name = "retrieval"
    description = "Manages and tunes the RAG pipeline"
    system_prompt = """You are the RAG (Retrieval-Augmented Generation) specialist of KONI-Forge.

Role:
- Report what is indexed in the vector store (collections, documents, chunk counts) and the current RAG configuration.
- Recommend chunking parameters (chunk size, overlap, splitting strategy) that fit the document type.
- Explain retrieval quality issues and how to improve them.
- Check whether KBD (Knowledge Boundary Detection) can run on the indexed documents.

After documents are indexed, always tell the user:
- "KBD is recommended next — it measures how much of these documents the base model already knows, and decides whether retrieval alone is enough or fine-tuning is needed."

Use the tools to look things up instead of guessing. Keep answers concise and concrete."""

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_vectorstore_status",
                "description": "Read the current state of the vector store (ChromaDB).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "list_indexed_documents",
                "description": "List the indexed documents.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_rag_config",
                "description": "Read the current RAG pipeline settings.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "check_kbd_readiness",
                "description": "Check whether Knowledge Boundary Detection can run. With documents indexed, it points to the KBD agent.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "recommend_generation_strategy",
                "description": "Recommend a QA generation strategy for the indexed documents, from document type, chunk count and content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document id to analyse (must be indexed)",
                        },
                    },
                    "required": ["doc_id"],
                },
            },
            {
                "name": "list_raw_documents",
                "description": "List the uploaded source documents (raw_dataset).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        if tool_name == "get_vectorstore_status":
            return self._get_vectorstore_status()
        elif tool_name == "list_indexed_documents":
            return self._list_indexed_documents()
        elif tool_name == "get_rag_config":
            return self._get_rag_config()
        elif tool_name == "check_kbd_readiness":
            return await self._check_kbd_readiness()
        elif tool_name == "recommend_generation_strategy":
            return self._recommend_generation_strategy(tool_input["doc_id"])
        elif tool_name == "list_raw_documents":
            return self._list_raw_documents()
        return f"Unknown tool: {tool_name}"

    def _get_vectorstore_status(self) -> str:
        """ChromaDB status check — uses the real persistence dir + live collection counts."""
        chroma_dir = Path(CHROMA_PERSIST_DIR)
        if not chroma_dir.exists():
            return json.dumps(
                {
                    "status": "not_initialized",
                    "message": self._rt("res_no_vectorstore"),
                    "suggestion": "Upload documents on the data page and they are indexed automatically.",
                },
                ensure_ascii=False,
            )

        collections = _list_chroma_collections()
        total_chunks = sum(c["chunk_count"] for c in collections)
        total_size = sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())
        return json.dumps(
            {
                "status": "active",
                "path": str(chroma_dir),
                "backend": "ChromaDB",
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "collection_count": len(collections),
                "total_chunks": total_chunks,
                "collections": collections[:20],  # cap preview
            },
            ensure_ascii=False,
        )

    def _list_indexed_documents(self) -> str:
        """Indexed document list — each entry is one ChromaDB collection (= one upload folder)."""
        collections = [c for c in _list_chroma_collections() if c["chunk_count"] > 0]
        if not collections:
            return json.dumps(
                {
                    "documents": [],
                    "message": self._rt("res_no_indexed_docs"),
                },
                ensure_ascii=False,
            )
        # Project to the shape downstream tools and the LLM expect
        docs = [
            {
                "id": c["name"],  # collection name == raw folder name
                "name": c["name"],
                "chunks": c["chunk_count"],
            }
            for c in collections
        ]
        return json.dumps(
            {
                "documents": docs,
                "total_collections": len(docs),
                "total_chunks": sum(d["chunks"] for d in docs),
            },
            ensure_ascii=False,
        )

    def _get_rag_config(self) -> str:
        """RAG configuration"""
        config_file = Path("/app/rag_config.json")
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

        # Default config
        return json.dumps(
            {
                "status": "not_configured",
                "defaults": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "embedding_model": "intfloat/multilingual-e5-large",
                    "top_k": 5,
                    "similarity_threshold": 0.7,
                    "self_rag_enabled": True,
                    "reranker": None,
                },
                "message": self._rt("res_default_settings"),
            },
            ensure_ascii=False,
            indent=2,
        )

    async def _check_kbd_readiness(self) -> str:
        """Check KBD analysis readiness by polling ChromaDB collections.

        - If a collection with at least 1 chunk exists → ready=True and emit
          a `__chain_signal__` so the dispatcher offers the next (KBD) stage.
        - Otherwise, if a Celery indexing job is active for some raw folder,
          wait briefly for it to complete.
        """
        import asyncio

        # Short busy-wait (up to ~60s) while an active Celery indexing job finishes
        async def _wait_for_active_job():
            try:
                from server.core.state import _get_redis, get_job

                r = _get_redis()
                for _ in range(12):
                    keys = r.keys(f"{RAG_JOB_KEY_PREFIX}*")
                    active = False
                    for k in keys:
                        jid = r.get(k)
                        if not jid:
                            continue
                        job = get_job(jid)
                        if job and job.get("status", "").upper() in ("STARTED", "PENDING", "RETRY"):
                            active = True
                            break
                    if not active:
                        return
                    await asyncio.sleep(5)
            except Exception:
                return

        collections = [c for c in _list_chroma_collections() if c["chunk_count"] > 0]
        if not collections:
            await _wait_for_active_job()
            collections = [c for c in _list_chroma_collections() if c["chunk_count"] > 0]

        if not collections:
            return json.dumps(
                {
                    "ready": False,
                    "message": self._rt("res_no_indexed_docs_data_page"),
                    "indexed_documents": [],
                },
                ensure_ascii=False,
            )

        # Build docs payload and pick a primary doc for chain metadata
        docs = [{"id": c["name"], "name": c["name"], "chunks": c["chunk_count"]} for c in collections]
        total_chunks = sum(d["chunks"] for d in docs)
        primary = max(docs, key=lambda d: d["chunks"])

        return json.dumps(
            {
                "ready": True,
                "message": _stage_text("kbd_ready", self._lang, collections=len(docs), chunks=total_chunks),
                "indexed_documents": docs,
                "next_step": "Run the KBD agent (ask the 'boundary' agent for an analysis)",
                "__chain_signal__": {
                    "stage_completed": "retrieval",
                    "metadata": {
                        "indexed_count": len(docs),
                        "total_chunks": total_chunks,
                        "doc_name": primary["name"],
                    },
                },
            },
            ensure_ascii=False,
        )

    def _recommend_generation_strategy(self, doc_id: str) -> str:
        """Recommend Q&A data generation strategy for one indexed collection.

        `doc_id` should be a ChromaDB collection name (equal to the raw folder
        name produced by /api/data/upload). If it's unknown, fall back to the
        largest available collection so the LLM can still proceed.
        """
        collections = [c for c in _list_chroma_collections() if c["chunk_count"] > 0]
        if not collections:
            return json.dumps({"error": _err("no_indexed_documents", self._lang)})

        target = next((c for c in collections if c["name"] == doc_id), None)
        if target is None:
            target = max(collections, key=lambda c: c["chunk_count"])

        chunk_count = target["chunk_count"]
        doc_name = target["name"]

        if chunk_count <= 10:
            qa_per_file, strategy_note = 5, "small document — focus on the key content"
        elif chunk_count <= 50:
            qa_per_file, strategy_note = 7, "medium document — balanced coverage"
        else:
            qa_per_file, strategy_note = 10, "large document — broad coverage"

        return json.dumps(
            {
                "doc_id": doc_name,
                "doc_name": doc_name,
                "chunk_count": chunk_count,
                "recommendation": {
                    "qa_per_file": qa_per_file,
                    "temperature": 0.7,
                    "strategy_note": strategy_note,
                    "message": (
                        _stage_text(
                            "qa_recommendation",
                            self._lang,
                            dataset=doc_name,
                            chunks=chunk_count,
                            qa_per_file=qa_per_file,
                        )
                    ),
                },
                "options": {
                    "option_1": "Use the recommendation — start generating with these parameters",
                    "option_2": "Configure manually — adjust the parameters first",
                },
            },
            ensure_ascii=False,
        )

    def _list_raw_documents(self) -> str:
        """Raw document list."""
        from server.core.corpus_target import is_document

        raw_dir = raw_root()
        if not raw_dir.exists():
            return json.dumps({"documents": [], "message": self._rt("res_no_raw_dir")}, ensure_ascii=False)

        docs = []
        for f in sorted(raw_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(raw_dir))
            if not is_document(rel):
                continue
            docs.append(
                {
                    "name": rel,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "type": f.suffix,
                }
            )
        return json.dumps({"documents": docs, "count": len(docs)}, ensure_ascii=False)
