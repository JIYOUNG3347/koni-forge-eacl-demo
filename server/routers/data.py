"""Data management router — /api/data

POST   /upload                    — file upload with MIME validation
GET    /datasets                  — 3-category listing (raw, processed, training)
DELETE /datasets/{dataset_id}     — delete dataset + related RAG indices
POST   /generate                  — dispatch QA generation Celery task
GET    /datasets/{dataset_id}/preview — preview dataset contents
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile

from server.core.celery_probe import active_task_ids
from server.core.classify_dataset_format import (
    classify_dataset_format,
    resolve_upload_target,
    write_dataset_format,
)
from server.core.dataset_files import read_dataset_format
from server.core.dataset_paths import (
    chroma_dir,
    corpus_root,
    dataset_output_root,
    find_dataset_dir,
    raw_root,
)
from server.core.dataset_versioning import parse_version
from server.core.helpers import safe_upload_basename
from server.core.logging import sys_log
from server.core.ttl_cache import TTLCache

router = APIRouter(prefix="/api/data", tags=["Data"])

RAW_DIR = raw_root()
TRAINING_DIR = corpus_root()

OUTPUT_DIR = dataset_output_root()

# Redis key mapping folder_name → latest indexing job id (TTL 2h)
_RAG_JOB_KEY_PREFIX = "rag_idx:"
_RAG_JOB_TTL = 60 * 60 * 2


# Redis key mapping source_folder → active preprocessing/TWIST job (TTL 4h)

# Short-TTL response cache for the GET /datasets listing. The handler is
# expensive (ChromaDB count per raw folder,
# disk stat across every dataset file). Polling at 5-15s with that cost
# saturated the uvicorn event loop. A 2s cache lets the polling cadence
# proceed without re-running the work each tick, while mutation routes
# invalidate so user-triggered changes show up immediately. Key includes
# role to stay correct if admin/user responses ever diverge.
_DATASETS_CACHE_KEY_PREFIX = "datasets_listing_cache:"
_DATASETS_CACHE_TTL = 2  # seconds


def _datasets_cache_key(user_id: str, role: str) -> str:
    return f"{_DATASETS_CACHE_KEY_PREFIX}{role or 'user'}:{user_id}"


def _invalidate_datasets_cache(user_id: str, role: str) -> None:
    """Drop the cached listing for this (role, user_id). Called from every
    mutation route so the next listing call recomputes."""
    try:
        from server.core.state import _get_redis

        _get_redis().delete(_datasets_cache_key(user_id, role))
    except Exception:
        pass  # cache miss is always safe; surface no error to the caller


def _invalidate_datasets_cache_for_request(request: Request) -> None:
    """Convenience wrapper for mutation routes — pulls user_id/role from
    request.state and drops the matching cache entry."""
    user_id = getattr(request.state, "user_id", "default")
    role = getattr(request.state, "user_role", "")
    _invalidate_datasets_cache(user_id, role)


def _set_indexing_job(folder_name: str, job_id: str) -> None:
    try:
        from server.core.state import _get_redis

        _get_redis().set(f"{_RAG_JOB_KEY_PREFIX}{folder_name}", job_id, ex=_RAG_JOB_TTL)
    except Exception as e:
        sys_log(f"[Data] Failed to persist rag_idx mapping: {e}", level="WARNING")


def _get_indexing_status(folder_name: str, chroma_client=None) -> Optional[dict]:
    """Return combined RAG indexing status for a raw-dataset folder.

    Checks ChromaDB collection count first (authoritative for 'completed'),
    then Redis job state (from Celery task updates) for in-progress info.
    Returns None when nothing is known about the folder.

    chroma_client: optional pre-created PersistentClient to reuse across calls.
    When provided, avoids creating a new SQLite connection per folder.
    When None, creates a client locally (fallback for standalone calls).
    """
    chunk_count = 0
    collection_exists = False
    client = chroma_client
    if client is None:
        try:
            import chromadb

            chroma_path = str(chroma_dir())
            client = chromadb.PersistentClient(path=chroma_path)
        except Exception:
            client = None
    if client is not None:
        try:
            col = client.get_collection(name=folder_name)
            chunk_count = col.count()
            collection_exists = True
        except Exception:
            collection_exists = False

    job_id = None
    job_data: Optional[dict] = None
    try:
        from server.core.state import _get_redis, get_job

        job_id = _get_redis().get(f"{_RAG_JOB_KEY_PREFIX}{folder_name}")
        if job_id:
            job_data = get_job(job_id)
    except Exception:
        pass

    job_status = (job_data or {}).get("status", "").upper() if job_data else ""

    # Decide final status: prefer Chroma count when collection has data.
    if collection_exists and chunk_count > 0:
        status = "completed"
    elif job_status in ("STARTED", "PENDING", "RETRY"):
        status = "indexing"
    elif job_status == "FAILURE":
        status = "failed"
    elif job_status == "SUCCESS":
        # Finished but Chroma empty — treat as completed (0 chunks)
        status = "completed"
    elif job_id:
        status = "queued"
    else:
        return None

    result: dict = {
        "status": status,
        "chunk_count": chunk_count,
        "job_id": job_id,
    }
    if job_data:
        try:
            result["progress"] = int(job_data.get("progress") or 0)
        except (ValueError, TypeError):
            result["progress"] = 0
        if job_data.get("message"):
            result["message"] = job_data["message"]
        if job_data.get("error"):
            result["error"] = job_data["error"]
        job_result = job_data.get("result")
        if isinstance(job_result, dict):
            if job_result.get("files_processed") is not None:
                result["files_processed"] = job_result["files_processed"]
            if job_result.get("total_chunks") is not None and not result["chunk_count"]:
                result["chunk_count"] = job_result["total_chunks"]
    return result


# Internal metadata / log files that should not be shown as user-facing data files.
# These still live inside the folder (useful for debugging / audit), but the
# dataset listing hides them so the user only sees the actual training payload.
HIDDEN_META_FILES = {
    "generation_params.json",
    "provenance.json",
    "preprocessing_stats.txt",
    "training_params.json",
    "generation.log",
    "preprocess.log",
    "twist.log",
    ".DS_Store",
}
HIDDEN_META_SUFFIXES = (".log",)
# Stem suffixes treated as refine audit sidecars (stats JSON, not data)
HIDDEN_META_STEM_SUFFIXES = ("_refined_stats",)

# Filenames considered the authoritative dataset payload when present.
DATA_FILE_NAMES = {"qa_dataset.json"}

from server.core.corpus_files import (  # noqa: E402
    select_data_files,
)
from server.core.dataset_file_view import (  # noqa: E402
    build_file_entries,
    remaining_after,
    removable_reason,
    training_ready,
)


def _upload_handoff(folder: str, target: str, user_id: str) -> Optional[str]:
    """Instruction handed to the agent after an upload.

    This text is read by the LLM rather than shown on screen, so it follows the
    pipeline's execution language, not the display language in the request header.

    ``target`` is the backend value (``raw`` for source documents), not the
    front-end name for the upload section.

    A failure to build it never rolls back the upload — ``None`` just means the
    front end sends no handoff, and the reason is logged.
    """
    try:
        from server.core.agent_handoff import handoff
        from server.core.lang import resolve_lang
        from server.core.pipeline_config import load_pipeline_config

        lang = resolve_lang(config=load_pipeline_config(user_id))
        if target == "raw":
            return handoff("upload_docs_next", lang, folder)
        return handoff("upload_qa_next", lang, folder)
    except Exception as exc:  # noqa: BLE001 — a handoff failure must not block the upload
        sys_log(f"[Data] upload handoff build failed (folder={folder}): {exc}", level="WARNING")
        return None


def _count_qa_records(file_path: Path) -> Optional[int]:
    """Best-effort QA record count for a dataset file. Returns None on error.

    Uses mtime-keyed LRU cache — if the file hasn't changed, skips the read.
    """
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        return None
    return _count_qa_records_cached(str(file_path), mtime)


@lru_cache(maxsize=256)
def _count_qa_records_cached(file_path_str: str, mtime: float) -> Optional[int]:
    """Cached implementation — mtime in the key ensures auto-invalidation on change."""
    file_path = Path(file_path_str)
    try:
        if file_path.suffix.lower() == ".jsonl":
            count = 0
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("qa_pairs", "items", "data", "records"):
                v = data.get(key)
                if isinstance(v, list):
                    return len(v)
    except Exception:
        return None
    return None


ALLOWED_EXTENSIONS = {".pdf", ".hwp", ".hwpx", ".docx", ".pptx", ".txt", ".md", ".json", ".jsonl"}
ALLOWED_MIMES = {
    "application/pdf",
    "application/x-hwp",
    "application/haansofthwp",
    "application/vnd.hancom.hwp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "application/x-ndjson",
    "application/octet-stream",  # fallback for HWP/HWPX
}

# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

_TARGET_ROOTS = {
    "raw": RAW_DIR,
    "processed": OUTPUT_DIR,
    "training": OUTPUT_DIR,
}
_TARGET_LABELS = {
    "raw": "Source documents",
    "processed": "QA dataset (training)",
    "training": "QA dataset (training)",
}


@router.post("/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    target: str = Form("raw"),
    folder_name: Optional[str] = Form(None),
):
    """Upload files into the requested dataset category.

    ``raw`` (default) takes source documents and queues RAG indexing;
    ``processed`` and ``training`` take a QA dataset as JSON or JSONL.
    """
    user_id = getattr(request.state, "user_id", "default")

    if target not in _TARGET_ROOTS:
        raise HTTPException(status_code=400, detail=f"Unknown upload target: {target}")

    # QA-style targets only accept json/jsonl — raw accepts the full ext set.
    is_qa_target = target in ("processed", "training")
    allowed_exts = {".json", ".jsonl"} if is_qa_target else ALLOWED_EXTENSIONS

    try:
        import magic
    except ImportError:
        magic = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"dataset_{timestamp}"

    saved: List[str] = []
    rejected: List[dict] = []
    incoming = list(files)

    buffered: List[tuple] = []  # (save_name, content)
    data_format: Optional[str] = None
    source_filename: Optional[str] = None
    for idx, f in enumerate(incoming):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in allowed_exts:
            rejected.append(
                {
                    "file": f.filename,
                    "reason": (
                        f"unsupported extension: {ext} "
                        + ("(QA target accepts only .json/.jsonl)" if is_qa_target else "")
                    ).strip(),
                }
            )
            continue

        content = await f.read()

        # MIME validation (raw target only — python-magic often mis-detects JSON)
        if not is_qa_target and magic is not None:
            detected_mime = magic.from_buffer(content, mime=True)
            if detected_mime not in ALLOWED_MIMES:
                rejected.append({"file": f.filename, "reason": f"MIME mismatch: {detected_mime}"})
                continue

        if is_qa_target and ext in (".json", ".jsonl"):
            fmt = classify_dataset_format(content, ext)
            if fmt == "preference":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "This looks like a preference dataset ({prompt, chosen, rejected}). "
                        "This edition trains SFT only — upload {question, answer} pairs."
                    ),
                )
            if fmt == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported data format. JSON and JSONL accept only SFT pairs: {question, answer}",
                )
            if data_format is None:
                data_format = fmt
                source_filename = f.filename

        # For QA targets with a single uploaded file, normalise the filename
        # to `qa_dataset.{ext}` so `scan_dir`'s DATA_FILE_NAMES lookup picks it
        # up and the training/refine pipelines find it without extra wiring.
        if is_qa_target and len(incoming) == 1:
            save_name = f"qa_dataset{ext}"
        else:
            # Strip any directory prefix (folder uploads send "subdir/file.pdf")
            # and reject path-traversal / empty names — defence in depth even
            # though the frontend now sends basenames.
            save_name = safe_upload_basename(f.filename) or f"upload_{idx}{ext}"

        buffered.append((save_name, content))

    if not buffered and rejected:
        raise HTTPException(status_code=400, detail={"message": "All files rejected", "rejected": rejected})

    effective_target = resolve_upload_target(target, data_format)
    dest_dir = _TARGET_ROOTS[effective_target] / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for save_name, content in buffered:
        (dest_dir / save_name).write_bytes(content)
        saved.append(save_name)

    if data_format in ("sft", "preference"):
        try:
            write_dataset_format(dest_dir, data_format, source_filename=source_filename)
        except Exception as _fe:
            sys_log(f"[Data] Failed to persist dataset format marker: {_fe}", level="WARNING")

    sys_log(
        f"[Data] Uploaded {len(saved)} files to {folder_name} "
        f"(target={effective_target}, format={data_format or '-'}, user={user_id})"
    )

    if folder_name:
        try:
            from server.core.provenance import write_upload_label

            write_upload_label(dest_dir, folder_name.strip())
        except Exception as _e:
            sys_log(f"[Data] Failed to persist upload folder label: {_e}", level="WARNING")

    # RAG indexing only makes sense for raw document uploads.
    indexing_job_id: Optional[str] = None
    if target == "raw":
        indexable_exts = {".pdf", ".hwp", ".hwpx", ".docx", ".pptx", ".txt", ".md"}
        indexable_saved = [s for s in saved if Path(s).suffix.lower() in indexable_exts]
        if indexable_saved:
            try:
                from celery_app.tasks.data_tasks import index_documents_task

                indexing_job_id = f"index-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex()}"
                index_documents_task.apply_async(
                    kwargs={
                        "job_id": indexing_job_id,
                        "user_id": user_id,
                        "dataset_folder": folder_name,
                        "collection_name": folder_name,
                        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "multilingual-e5-large"),
                        "chunk_size": 512,
                        "chunk_overlap": 50,
                    },
                    task_id=indexing_job_id,
                )
                _set_indexing_job(folder_name, indexing_job_id)
                sys_log(
                    f"[Data] Auto-queued RAG indexing: {indexing_job_id} "
                    f"(folder={folder_name}, files={len(indexable_saved)}, user={user_id})"
                )
            except Exception as e:
                sys_log(f"[Data] Auto-index dispatch failed: {e}", level="WARNING")
                indexing_job_id = None

    _invalidate_datasets_cache_for_request(request)
    return {
        "folder": folder_name,
        "target": effective_target,
        "target_label": _TARGET_LABELS[effective_target],
        "saved": saved,
        "rejected": rejected,
        "total": len(saved),
        "indexing_job_id": indexing_job_id,
        "data_format": data_format,
        "agent_handoff": _upload_handoff(folder_name, effective_target, user_id),
    }


# ---------------------------------------------------------------------------
# GET /datasets
# ---------------------------------------------------------------------------


@router.get("/datasets")
async def list_datasets(request: Request):
    """List datasets in 3 categories: raw, processed, training."""
    # The full handler costs N+1 Redis keys, a ChromaDB count per dataset and a
    # disk stat, which the UI's 5-15s polling would repeat on every tick. A 2s
    # cache absorbs that; mutation routes invalidate it.
    user_id = getattr(request.state, "user_id", "default")
    role = getattr(request.state, "user_role", "")
    cache_key = _datasets_cache_key(user_id, role)
    _r = None
    try:
        from server.core.state import _get_redis as _get_redis_for_cache

        _r = _get_redis_for_cache()
        cached = _r.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        _r = None  # cache failure → skip cache write at the bottom too

    # D1: Create one ChromaDB PersistentClient for the whole listing operation.
    # Each _get_indexing_status call reuses it instead of opening a new SQLite
    # connection per raw folder — eliminates N×WAL-lock contention during indexing.
    _chroma_client = None
    try:
        import chromadb as _chromadb

        _chroma_path = str(chroma_dir())
        _chroma_client = _chromadb.PersistentClient(path=_chroma_path)
    except Exception:
        pass  # fall through — _get_indexing_status will handle None gracefully

    def scan_dir(base: Path, category: str):
        results = []
        if not base.exists():
            return results
        for entry in sorted(base.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            all_files = [f for f in entry.iterdir() if f.is_file()]
            file_names = [f.name for f in all_files]
            _sizes = {f.name: f.stat().st_size for f in all_files}
            has_output = any(n in DATA_FILE_NAMES or n.endswith(".jsonl") for n in file_names)
            if category != "raw" and not has_output:
                continue

            # File visibility: hide metadata / log files.
            def _is_hidden(n: str) -> bool:
                if n in HIDDEN_META_FILES:
                    return True
                if any(n.endswith(suf) for suf in HIDDEN_META_SUFFIXES):
                    return True
                stem = Path(n).stem
                if any(stem.endswith(suf) for suf in HIDDEN_META_STEM_SUFFIXES):
                    return True
                return False

            display_files = [n for n in file_names if not _is_hidden(n)]
            total_size = sum(f.stat().st_size for f in all_files)

            qa_count: Optional[int] = None
            _selected = select_data_files(file_names)
            if _selected:
                qa_count = _count_qa_records(entry / _selected[0])

            _base_id, _version = parse_version(entry.name)
            item = {
                "id": entry.name,
                "category": category,
                "name": entry.name,
                "base_id": _base_id,
                "version": _version,
                "file_count": len(display_files),
                "files": display_files[:50],  # cap preview
                "file_entries": build_file_entries(file_names, _sizes)[:50],
                "size_mb": round(total_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
                "qa_count": qa_count,
                "data_format": read_dataset_format(entry),
            }
            if category == "raw":
                idx_status = _get_indexing_status(entry.name, chroma_client=_chroma_client)
                if idx_status is not None:
                    item["indexing"] = idx_status
            results.append(item)
        return results

    raw = scan_dir(RAW_DIR, "raw")
    training = scan_dir(TRAINING_DIR, "training")

    _indexing_items = [
        item for item in raw if isinstance(item.get("indexing"), dict) and item["indexing"].get("status") == "indexing"
    ]
    if _indexing_items:
        _celery_active_ids: Optional[set] = await asyncio.to_thread(active_task_ids)

        if _celery_active_ids is not None:
            for _item in _indexing_items:
                _idx = _item.get("indexing", {})
                _job_id = _idx.get("job_id")
                if _job_id and _job_id not in _celery_active_ids:
                    _folder = _item["id"]
                    sys_log(f"[Data] Zombie indexing job detected, auto-cleaning: {_folder}")
                    try:
                        from server.core.state import update_job as _update_job

                        _update_job(_job_id, status="FAILURE", error="Worker terminated unexpectedly")
                    except Exception:
                        pass
                    _item["indexing"]["status"] = "failed"  # mutate response in-place

    response = {
        "raw": raw,
        "training": training,
        "total": len(raw) + len(training),
    }
    # Cache the computed listing. Skip if the earlier cache GET couldn't
    # reach Redis (_r stays None), or swallow any setex error — the response
    # has already been built and failing to cache just means the next poll
    # recomputes (the original behaviour).
    if _r is not None:
        try:
            _r.setex(cache_key, _DATASETS_CACHE_TTL, json.dumps(response, ensure_ascii=False))
        except Exception:
            pass
    return response


# ---------------------------------------------------------------------------
# DELETE /datasets/{dataset_id}
# ---------------------------------------------------------------------------

# Category -> storage base. `?category=` says which copy of a same-named
# dataset to delete.
_CATEGORY_BASES = {
    "raw": RAW_DIR,
    "training": TRAINING_DIR,
}


def _cleanup_rag_state(dataset_id: str) -> Tuple[bool, int]:
    """Delete the per-folder ChromaDB collection + Redis rag_idx:{folder}
    key for a raw dataset. Returns (chroma_cleaned, redis_keys_cleared).

    Silent on missing resources — never raises. The collection might not
    exist (dataset never indexed) and the Redis key has a TTL; either
    being absent is normal cleanup. Errors are logged at WARNING for
    visibility but don't fail the delete request.

    Indexing writes one collection per source folder, so deleting by
    dataset_id targets the right collection with no ambiguity.
    """
    chroma_cleaned = False
    try:
        import chromadb

        chroma_path = str(chroma_dir())
        client = chromadb.PersistentClient(path=chroma_path)
        client.delete_collection(name=dataset_id)
        chroma_cleaned = True
        sys_log(f"[Data] Deleted ChromaDB collection '{dataset_id}'")
    except Exception as e:
        # Most common: collection never existed (dataset wasn't indexed).
        # Logged but not surfaced — same outcome from the user's POV.
        sys_log(f"[Data] ChromaDB collection cleanup skipped for '{dataset_id}': {e}")

    redis_keys_cleared = 0
    try:
        from server.core.state import _get_redis

        r = _get_redis()
        if r.delete(f"{_RAG_JOB_KEY_PREFIX}{dataset_id}"):
            redis_keys_cleared += 1
    except Exception:
        pass

    return chroma_cleaned, redis_keys_cleared


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    request: Request,
    category: Optional[str] = Query(
        None,
        regex="^(raw|training)$",
        description=(
            "Storage category to delete from: raw / training. "
            "When omitted, falls back to legacy first-match behaviour "
            "which is buggy for auto-promoted datasets — frontend should "
            "always pass category."
        ),
    ),
):
    """Delete a dataset from a specific storage category.

    For raw deletions, the per-folder ChromaDB collection and the
    `rag_idx:{folder}` Redis key are also cleaned up — they're tied to
    the raw upload's lifecycle. processed / training deletions leave RAG
    state alone (those categories don't own it).

    """
    user_id = getattr(request.state, "user_id", "default")
    warnings: List[str] = []

    # ── Legacy fallback: no category provided ─────────────────────────
    # Preserved so internal scripts and any caller that hasn't been
    # updated keep working. The loop+break bug stays in this branch on
    # purpose — fixing it without category info is impossible (we can't
    # guess which copy the caller meant). The warning makes it visible
    # if anything's still using this path.
    if category is None:
        sys_log(
            f"[Data] DELETE /datasets/{dataset_id} called without "
            "?category= — using legacy first-match path. This is buggy "
            "for auto-promoted datasets; callers should pass category.",
            level="WARNING",
        )
        deleted_from = None
        for cat_name, base in _CATEGORY_BASES.items():
            target = base / dataset_id
            if target.exists() and target.is_dir():
                try:
                    shutil.rmtree(target)
                except Exception as e:
                    sys_log(
                        f"[Data] rmtree failed for '{dataset_id}' in {cat_name}: {e}",
                        level="WARNING",
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to remove '{dataset_id}' from {cat_name}: {e}",
                    )
                deleted_from = cat_name
                break
        if deleted_from is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        if deleted_from == "raw":
            _cleanup_rag_state(dataset_id)
        sys_log(f"[Data] Deleted dataset '{dataset_id}' from {deleted_from} (legacy, user={user_id})")
        _invalidate_datasets_cache_for_request(request)
        return {
            "message": f"Dataset '{dataset_id}' deleted",
            "dataset_id": dataset_id,
            "category": deleted_from,
            "legacy_fallback": True,
        }

    # ── Category-aware path ───────────────────────────────────────────
    base = _CATEGORY_BASES[category]
    target = base / dataset_id

    if not target.exists() or not target.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found in category '{category}'",
        )

    # Indexing-in-progress guard (raw only). Don't refuse the delete —
    # the user explicitly asked — but skip RAG cleanup so we don't race
    # the worker writing to the same Chroma collection. The warning
    # surfaces in the response so the UI can advise cancelling the
    # indexing job first for a fully clean state.
    skip_rag_cleanup = category != "raw"
    if category == "raw":
        try:
            idx = _get_indexing_status(dataset_id)
        except Exception:
            idx = None
        if idx and idx.get("status") in ("indexing", "queued"):
            warnings.append(
                f"Indexing job is in progress for '{dataset_id}'. The "
                "folder was removed but ChromaDB / Redis RAG state was "
                "left intact to avoid racing the worker. Cancel the "
                "indexing job and delete again for a fully clean state."
            )
            skip_rag_cleanup = True

    # Disk delete — graceful on rmtree errors so the request reports
    # partial state instead of returning 500 with no detail.
    try:
        shutil.rmtree(target)
    except Exception as e:
        sys_log(
            f"[Data] rmtree failed for '{dataset_id}' ({category}): {e}",
            level="WARNING",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove '{dataset_id}' from {category}: {e}",
        )

    # RAG state cleanup — raw only, and only when indexing isn't active.
    chroma_cleaned = False
    rag_redis_cleared = 0
    if not skip_rag_cleanup:
        chroma_cleaned, rag_redis_cleared = _cleanup_rag_state(dataset_id)

    sys_log(
        f"[Data] Deleted dataset '{dataset_id}' from {category} "
        f"(user={user_id}, chroma_cleaned={chroma_cleaned}, "
        f"rag_redis_cleared={rag_redis_cleared}, "
        f")"
    )
    _invalidate_datasets_cache_for_request(request)
    return {
        "message": f"Dataset '{dataset_id}' deleted from {category}",
        "dataset_id": dataset_id,
        "category": category,
        "chroma_cleaned": chroma_cleaned,
        "rag_redis_cleared": rag_redis_cleared,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# DELETE /datasets/{dataset_id}/files/{file_name}
# ---------------------------------------------------------------------------


@router.delete("/datasets/{dataset_id}/files/{file_name}")
async def delete_dataset_file(dataset_id: str, file_name: str, request: Request):
    """**Deleting the last training file is allowed.** The opposite of "undo the
    augmentation" is "undo the preprocessing too", which is a legitimate intent.
    The outcome is reported as ``training_ready: false`` instead of blocked

    """
    user_id = getattr(request.state, "user_id", "default")

    # Normalise the name before building the path — `Path(...).name` strips
    # both directory components and `../` (the same guard the upload path uses).
    safe_name = safe_upload_basename(file_name)
    directory = find_dataset_dir(dataset_id)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    file_names = [f.name for f in directory.iterdir() if f.is_file()]
    reason = removable_reason(file_names, safe_name or "")
    if reason == "not_found":
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in '{dataset_id}'")
    if reason is not None:
        raise HTTPException(status_code=400, detail=f"Invalid file name: {file_name}")

    target = directory / safe_name
    # Even after normalisation, anything pointing outside the folder is refused.
    if not target.is_file() or target.resolve().parent != directory.resolve():
        raise HTTPException(status_code=400, detail=f"Invalid file name: {file_name}")

    try:
        target.unlink()
    except OSError as exc:
        sys_log(f"[Data] Failed to delete {target}: {exc}", level="WARNING")
        raise HTTPException(status_code=500, detail=f"Failed to delete '{file_name}': {exc}")

    remaining = remaining_after(file_names, safe_name)
    _remaining_sizes = {f.name: f.stat().st_size for f in directory.iterdir() if f.is_file()}
    still_trainable = training_ready(remaining)
    _msg = f"[Data] Deleted file '{safe_name}' from '{dataset_id}'"
    sys_log(f"{_msg} (user={user_id}, training_ready={still_trainable})")
    _invalidate_datasets_cache_for_request(request)
    return {
        "dataset": dataset_id,
        "deleted": safe_name,
        "training_ready": still_trainable,
        "file_entries": build_file_entries(remaining, _remaining_sizes),
    }


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/preview
# ---------------------------------------------------------------------------


def _pick_preview_file(target: Path) -> Optional[Path]:
    """Pick the authoritative data file in `target` to preview.

    Order: qa_dataset.json > qa_dataset.jsonl > any non-metadata json/jsonl.
    """
    files = [f for f in target.iterdir() if f.is_file()]
    # preferred canonical names
    for name in ("qa_dataset.json", "qa_dataset.jsonl"):
        p = target / name
        if p.exists() and p.is_file():
            return p
    # 3) any non-metadata json/jsonl
    for f in sorted(files):
        if f.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if f.name in HIDDEN_META_FILES:
            continue
        stem = f.stem
        if any(stem.endswith(suf) for suf in HIDDEN_META_STEM_SUFFIXES):
            continue
        return f
    return None


def _load_records(path: Path) -> List[dict]:
    """Load QA records from a .json or .jsonl file; tolerant of nested shapes."""
    try:
        if path.suffix.lower() == ".jsonl":
            out: List[dict] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return out
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in ("qa_pairs", "items", "data", "records"):
                v = data.get(key)
                if isinstance(v, list):
                    return [r for r in v if isinstance(r, dict)]
            return [data]
    except Exception:
        pass
    return []


def _messages_to_qa(rec: dict) -> dict:
    """Project a record into a uniform {question, answer, ...} shape for UI."""
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs:
        q = a = ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "user" and not q:
                q = content
            elif role == "assistant" and not a:
                a = content
        return {**rec, "question": rec.get("question") or q, "answer": rec.get("answer") or a}
    return rec


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, request: Request, max_items: int = 20):
    """Preview dataset contents.

    * Raw folders  → list source document filenames (PDF/HWP/TXT upload).
    * Processed / Training folders → return QA records from the authoritative
      data file. Metadata files (`generation_params.json`,
      `preprocessing_stats.txt`, `*_refined_stats.json`, logs) are never
      surfaced. If a `_refined` payload exists, it is shown; otherwise the
      original `qa_dataset.json` is used.
    """
    target = None
    category = None
    for base, cat in [(TRAINING_DIR, "training"), (RAW_DIR, "raw")]:
        candidate = base / dataset_id
        if candidate.exists():
            target = candidate
            category = cat
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    # Raw datasets are source documents (PDF/HWP/...), not JSON records —
    # list filenames so the UI can show "which files were uploaded".
    if category == "raw":
        files = sorted(
            [f for f in target.iterdir() if f.is_file() and f.name not in HIDDEN_META_FILES],
            key=lambda f: f.name,
        )
        return {
            "dataset_id": dataset_id,
            "category": category,
            "items": [],  # no QA items for raw
            "total": 0,
            "source_files": [{"filename": f.name, "size": f.stat().st_size} for f in files[:max_items]],
        }

    # Processed / training: surface QA records from the single authoritative file.
    data_file = _pick_preview_file(target)
    if data_file is None:
        return {
            "dataset_id": dataset_id,
            "category": category,
            "items": [],
            "total": 0,
            "source_file": None,
            "message": "No data files yet.",
        }

    records = _load_records(data_file)
    projected = [_messages_to_qa(r) for r in records[:max_items]]

    return {
        "dataset_id": dataset_id,
        "category": category,
        "items": projected,
        "total": len(records),
        "source_file": data_file.name,
    }


@router.get("/datasets/{dataset_id}/provenance")
async def get_dataset_provenance(dataset_id: str, request: Request):
    """Datasets generated before provenance tracking have no sidecar — those
    return ``{"available": False, "reason": "pre-v2"}`` rather than 404, so the
    UI can distinguish "no lineage recorded" from "dataset missing".

    """
    from server.core.provenance import read_provenance

    target = None
    for base in (TRAINING_DIR, RAW_DIR):
        candidate = base / dataset_id
        if candidate.exists():
            target = candidate
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    record = read_provenance(target)
    if record is None:
        return {"dataset_id": dataset_id, "available": False, "reason": "pre-v2"}
    return {
        "dataset_id": dataset_id,
        "available": True,
        "provenance": record.to_dict(),
    }


_CHUNKS_CACHE = TTLCache(ttl_s=10.0)


@router.get("/datasets/{dataset_id}/chunks")
async def list_dataset_chunks(dataset_id: str, request: Request, limit: int = 20, offset: int = 0):
    """Read chunks from the Chroma collection (name=dataset_id) and return
    `chunk_index`, `page_num`, `doc_name` and a truncated `text` preview. An
    unindexed dataset gives an empty list; older chunks have `page_num=null`.

    """
    from server.core.chunk_meta import build_chunks_listing

    def _fetch_records():
        try:
            from modules.rag.vectorstore import VectorStore

            return VectorStore(collection_name=dataset_id).list_chunks()
        except Exception as e:
            sys_log(f"[Data] chunk listing fetch failed for {dataset_id}: {e}", level="WARNING")
            return []

    records = _CHUNKS_CACHE.get_or_compute(dataset_id, _fetch_records)
    result = build_chunks_listing(records, limit=limit, offset=offset)
    return {"dataset_id": dataset_id, **result}


_STORAGE_INFO_CACHE = TTLCache(ttl_s=10.0)


@router.get("/datasets/{dataset_id}/storage")
async def get_dataset_storage(dataset_id: str, request: Request):
    from server.core.storage_info import compute_storage_info

    target = None
    for base in (TRAINING_DIR, RAW_DIR):
        candidate = base / dataset_id
        if candidate.exists():
            target = candidate
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    def _sig():
        try:
            return target.stat().st_mtime
        except OSError:
            return None

    info = _STORAGE_INFO_CACHE.get_or_compute(dataset_id, lambda: compute_storage_info(target), signature=_sig)
    return {"dataset_id": dataset_id, **info}


_DATASET_FILES_CACHE = TTLCache(ttl_s=10.0)


@router.get("/datasets/{dataset_id}/files")
async def get_dataset_files(dataset_id: str, request: Request):
    """Source document files in a dataset folder (name, size, modified time).

    Complements provenance: a raw-stage dataset with no provenance still lists
    the source documents actually present in the folder (PDF, HWP, DOCX, PPTX,
    TXT, MD). QA payloads, metadata sidecars and `.koni_meta/` are excluded.
    """
    from server.core.dataset_files import list_dataset_files

    target = None
    for base in (TRAINING_DIR, RAW_DIR):
        candidate = base / dataset_id
        if candidate.exists():
            target = candidate
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    def _sig():
        try:
            return target.stat().st_mtime
        except OSError:
            return None

    files = _DATASET_FILES_CACHE.get_or_compute(dataset_id, lambda: list_dataset_files(target), signature=_sig)
    return {"dataset_id": dataset_id, "files": files, "total": len(files)}


