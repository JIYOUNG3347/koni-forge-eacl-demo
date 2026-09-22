"""Data tasks — document indexing on the celery ``default`` queue."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from server.core.config import (
    INTERNAL_API_URL,
)
from server.core.dataset_paths import raw_root
from server.core.logging import sys_log
from server.core.state import update_job


@shared_task(
    name="celery_app.tasks.data_tasks.index_documents_task",
    bind=True,
    soft_time_limit=int(os.getenv("INDEX_TIMEOUT_SECONDS", "1800")),
    time_limit=int(os.getenv("INDEX_TIMEOUT_SECONDS", "1800")) + 300,
    max_retries=0,
)
def index_documents_task(self, **params) -> Dict[str, Any]:
    """
    Index documents into ChromaDB for RAG.

    Worker does NOT access ChromaDB directly — it sends indexing
    requests through the app's /api/rag endpoint.

    Args (in params):
        job_id: Job identifier
        user_id: User who initiated indexing
        source_path: Path to documents
        collection_name: ChromaDB collection name
        chunk_size: Text chunk size
        chunk_overlap: Chunk overlap size
    """
    job_id = params.get("job_id", self.request.id or "unknown")
    # Support both 'source_path' (direct) and 'dataset_folder' (from router)
    dataset_folder = params.get("dataset_folder", "")
    source_path = params.get("source_path") or str(
        raw_root() / dataset_folder
    )
    collection_name = params.get("collection_name") or dataset_folder or "koni_documents"
    chunk_size = params.get("chunk_size", 512)
    chunk_overlap = params.get("chunk_overlap", 50)
    embedding_model = params.get("embedding_model", os.getenv("RAG_EMBEDDING_MODEL", "multilingual-e5-large"))

    sys_log(f"[index_docs] Starting job {job_id}: source={source_path}")
    update_job(job_id, status="STARTED", progress=0, message="Document indexing starting")

    try:
        import httpx

        # Read documents from source
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        from server.core.rag_index_plan import INDEXABLE_EXTS

        if source.is_file():
            files = [source]
        else:
            files = sorted([f for f in source.iterdir() if f.is_file() and f.suffix.lower() in INDEXABLE_EXTS])

        if not files:
            raise ValueError(f"No indexable files found in {source_path}")

        total_chunks = 0
        total_files = len(files)

        from server.core.chunk_meta import build_page_chunks
        from server.core.document_text import extract_pages as _extract_pages

        execute_url = f"{INTERNAL_API_URL}/api/rag/index/execute"
        # The first request also downloads the embedding model (about 1GB), which
        # can take several minutes. A short timeout here aborts the worker side
        # while the app finishes the write, so the chunk count is lost even
        # though the documents were indexed.
        execute_timeout = float(os.getenv("INDEX_REQUEST_TIMEOUT_SECONDS", "900"))

        for idx, filepath in enumerate(files):
            try:
                pages = _extract_pages(filepath)
                if not any((t or "").strip() for _, t in pages):
                    sys_log(f"[index_docs] Empty text from {filepath.name}, skipping", level="WARNING")
                    continue

                records = build_page_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not records:
                    continue
                chunks = [r["text"] for r in records]
                page_nums = [r["page_num"] for r in records]

                # Send chunks to app's /api/rag/index/execute (ChromaDB write)
                doc_id = f"{collection_name}_{filepath.stem}_{idx}"
                resp = httpx.post(
                    execute_url,
                    json={
                        "doc_id": doc_id,
                        "doc_name": filepath.name,
                        "chunks": chunks,
                        "page_nums": page_nums,
                        "collection_name": collection_name,
                        "embedding_model": embedding_model,
                        "metadata": {"source_folder": dataset_folder},
                    },
                    timeout=execute_timeout,
                )
                if resp.status_code == 200:
                    result_data = resp.json()
                    total_chunks += result_data.get("chunks_added", len(chunks))
                    sys_log(f"[index_docs] Indexed {filepath.name}: {result_data.get('chunks_added', 0)} chunks")
                else:
                    sys_log(
                        f"[index_docs] App API error for {filepath.name}: {resp.status_code} {resp.text[:200]}",
                        level="WARNING",
                    )

                progress = int(((idx + 1) / total_files) * 90)
                update_job(
                    job_id,
                    progress=progress,
                    message=f"Indexing {idx + 1}/{total_files}: {filepath.name} ({len(chunks)} chunks)",
                )

            except Exception as e:
                sys_log(
                    f"[index_docs] Failed to process {filepath.name}: {e}",
                    level="WARNING",
                )

        update_job(
            job_id,
            status="SUCCESS",
            progress=100,
            message="Document indexing completed",
            result={
                "files_processed": total_files,
                "total_chunks": total_chunks,
                "collection_name": collection_name,
            },
        )
        sys_log(f"[index_docs] Job {job_id} completed: {total_files} files, {total_chunks} chunks")
        return {
            "files_processed": total_files,
            "total_chunks": total_chunks,
            "collection_name": collection_name,
        }

    except SoftTimeLimitExceeded:
        update_job(job_id, status="FAILURE", error="Document indexing timed out")
        raise

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        update_job(job_id, status="FAILURE", error=error_msg)
        sys_log(f"[index_docs] Job {job_id} failed: {error_msg}", level="ERROR")
        raise
