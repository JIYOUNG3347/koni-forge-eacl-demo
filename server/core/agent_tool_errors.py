"""Agent tool error messages."""

from __future__ import annotations

from typing import Dict, Optional

from server.core.lang import EN

#: Tool error messages. Values may contain ``str.format`` placeholders.
TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        # ── tuning_specialist ──
        "gpu_query_failed": "Failed to query GPU information: {error}",
        "job_history_failed": "Failed to query job history: {error}",
        "log_read_failed": "Failed to read the log: {error}",
        "log_file_not_found": "Log file not found: {task_id}",
        "train_start_http_failed": "Failed to start training (HTTP {status}): {detail}",
        "train_start_failed": "Failed to start training: {error}",
        "train_no_job_id": "No job_id was returned after training started.",
        "train_job_ended": "The training job {status}: {detail}",
        "train_timeout": "Training timed out (12 hours). Check the status in the job list.",
        "dataset_not_found": "Dataset not found: {dataset}",
        # ── retrieval_specialist ──
        "no_indexed_documents": "There are no indexed documents.",
        "download_already_running": "Another download is already running. Try again once it finishes.",
        "download_failed": "The download failed: {detail}",
        "unknown_error": "unknown error",
        "download_timeout": "The download timed out ({minutes} minutes).",
        "job_name_exists": "That job name already exists: {job_name}",
        "download_start_http_failed": "Failed to start the download ({status}): {detail}",
        "download_no_job_id": "No job_id was returned for the download.",
        "index_list_failed": "Failed to list indexed documents: {error}",
        "collection_not_found": (
            "Collection '{doc_id}' was not found. Use get_indexed_documents to find a valid doc_id."
        ),
        "chunk_query_failed": "Failed to fetch chunks: {error}",
        "chunks_empty": "The document has no chunks. Check the doc_id.",
        "concept_parse_failed": "The key-concept extraction result could not be parsed.",
        "concept_extract_failed": "Key-concept extraction failed: {error}",
        "no_probe_questions": "There are no probe questions.",
        "model_name_empty": "model_name is empty.",
        "probe_dispatch_http_failed": "Probe dispatch failed: {status} {detail}",
        "probe_no_job_id": "No job_id was returned for the probe.",
        "probe_dispatch_failed": "Probe dispatch failed: {error}",
        "probe_poll_retry_exceeded": "Probe polling failed (connection retries exhausted): {error}",
        "probe_run_failed": "The probe run failed: {error}",
        "probe_poll_timeout": "Probe polling timed out ({seconds}s).",
        "probe_poll_failed": "Probe polling failed: {error}",
        "no_probe_results": "There are no probe results to score.",
        "no_scored_results": "There are no scored results.",
        "model_list_failed": "Failed to list models: {error}",
    },
}


def texts(lang: Optional[str] = None) -> Dict[str, str]:
    """Phrase pack for a language."""
    return TEXTS[EN]


def error_text(name: str, lang: Optional[str] = None, **fields: object) -> str:
    """One tool error message. **Never raises.**"""
    template = texts(lang).get(name)
    if template is None:
        return name
    if not fields:
        return template
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return template
