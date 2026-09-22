"""Runtime phrases written by the code, not the LLM.

This is behaviour, not just display: the loop-breaking message is appended to
the conversation as an assistant message and becomes part of the next turn's
LLM input.
"""

from __future__ import annotations

from typing import Dict, Optional

from server.core.lang import EN

TOOL_LABEL_KEYS: Dict[str, str] = {
    "start_training_job": "tool_label_training",
    "download_model": "tool_label_model_download",
}

TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        "analyzing": "Analysing...",
        "tool_loop_stopped": (
            "Stopping — the same tool was called repeatedly. Check the previous result and move on to the next step."
        ),
        "max_depth_reached": ("The pipeline reached its maximum depth. Continue with the next step manually."),
        "announce_training_params": "Recommended training parameters → start training with these?",
        "train_dataset_not_found": (
            "The requested dataset was not found: {mentioned}\n\n"
            "Available training datasets:\n{preview}\n\n"
            "Please ask again with one of the names above. Training was not started."
        ),
        "train_dataset_ambiguous": (
            "Several datasets were mentioned: {mentioned}\n\n"
            "Please name the one dataset to train on. Training was not started.\n{preview}"
        ),
        "train_missing_params": (
            "start_training_job is missing required parameters: {fields}. "
            "The exact values are in the instruction message earlier in this conversation — "
            "call the tool again including {fields}."
        ),
        "tool_call": "Tool call: {tool}",
        "agent_error": "Agent error: {error}",
        "agent_not_found_named": "The '{agent}' agent was not found.",
        "stage_skipped": "Skipped '{stage}'.",
        "model_downloading": "Downloading the {model} model...",
        "download_failed": "Download failed: {error}",
        "download_done": "{model} downloaded!",
        "download_error": "Download error: {error}",
        "kbd_pipeline_error": "KBD pipeline error: {error}",
        "kbd_resume_with_model": "Resuming the KBD analysis with the {model} model.",
        "kbd_start": "Starting the KBD analysis...",
        "kbd_no_indexed_docs": "No documents are indexed. Upload documents on the RAG page first.",
        "kbd_no_base_model": "No base model is available.",
        "kbd_run_with_model": "Running the KBD analysis with the {model} model.",
        "kbd_concepts_failed": "Could not extract key concepts: {error}",
        "kbd_probing": "Running {count} probe questions against the {model} model...",
        "kbd_probe_failed": "Probe run failed: {error}",
        "kbd_scoring": "Scoring {count} probe responses...",
        "unknown_error": "unknown error",
        "next_score_probes_with_results": "Call score_probe_responses, passing this result's results array as probe_results",
        "next_list_local_models": "Call list_local_models",
        "next_run_probes": "Call run_knowledge_probes, passing this result's probes array as probes and the chosen model name as model_name",
        "next_boundary_map": "Call generate_boundary_map(model_name='{model}', doc_name='{doc}'). scored_results may be left empty — the cached results are used automatically.",
        "next_list_training_datasets": "Call list_training_datasets to see the available training data.",
        "next_start_training": "Call start_training_job to start training. It needs the dataset, model_name and job_name parameters.",
        "next_upload_training_data": "There is nothing to train on. Tell the user to upload a QA dataset (JSON or JSONL) on the data page, and do not call start_training_job.",
        "res_no_training_dataset": "\n\n---\nThere is no training dataset yet. Upload a QA dataset (JSON or JSONL) on the data page, then ask for the training parameters again.\n",
        "train_dataset_unresolved": (
            "\n\n---\nThis run did not determine which dataset to train on, so nothing was pre-selected.\n\n"
            "Available training datasets:\n{preview}\n\n"
            "Name the one to use, or upload a QA dataset for this document. Training was not started.\n"
        ),
        "res_no_raw_dir": "The source document directory does not exist.",
        "res_no_vectorstore": "The vector store is not initialised yet.",
        "res_no_vectorstore_upload": "The vector store is not initialised yet. Upload documents on the data page.",
        "res_no_indexed_docs": "No documents are indexed.",
        "res_no_indexed_docs_upload": "No documents are indexed. Upload documents on the data page.",
        "res_no_indexed_docs_data_page": "No documents are indexed. Upload documents on the data page first.",
        "res_default_settings": "These are the defaults. You can customise them on the RAG page.",
        "res_no_job_history": "No job history.",
        "res_model_download_done": "The model '{model}' finished downloading.",
        "res_training_done": "The training job '{job}' finished!",
        "next_step_heading": "\n\n---\n**Suggested next step**: {announce}\n",
        "chain_skipped_end": "Skipped '{announce}'. There is no further step to run, so the pipeline ends here.",
        "resume_question": "I read that as a question. The '{label}' step is currently waiting.\nSay 'go' to proceed, or name another step.",
        "resume_unmatched": "I could not match that to a step. The '{label}' step is currently waiting.\nEnter one of: {options}",
        "resume_mismatched": "That is out of order — the suggested step is '{label}' but you asked for '{requested}'.\nTo run '{requested}' anyway, enter '{requested}' once more.",
        "tool_confirm_ask": "Proceed with {tool_label}?",
        "announce_skip_to_training": "Shall I start training now?",
        "stage_kbd_heading": "**KBD analysis results**\n",
        "stage_kbd_coverage": "- Knowledge coverage: **{value}%**",
        "stage_kbd_hallucination": "- Hallucination rate: **{value}%**",
        "stage_kbd_verdict_row": "- Verdict: **{value}**",
        "stage_kbd_weak": "- Weak categories: {categories}",
        "stage_kbd_boundary": "- Boundary categories: {categories}",
        "stage_kbd_verdict_rag": "The model already holds enough knowledge — **RAG alone can serve this**.",
        "stage_kbd_verdict_ft_low": "Knowledge coverage is very low — **Full Fine-Tuning (FFT)** is required.",
        "stage_kbd_verdict_ft": "Knowledge coverage is insufficient — **Fine-Tuning** is required.",
        "stage_kbd_verdict_hybrid": "**RAG + lightweight Fine-Tuning (LoRA)** is recommended.",
        "stage_kbd_verdict_other": "Verdict: {value}",
        "stage_tuning_done": "**Training complete** — model `{model}` ({method})",
        "small_data_note": (
            "⚠️ The dataset has only {samples} items, so your settings (epochs "
            "{before_epochs}, batch {before_batch}) would give just {estimated} training "
            "steps — adjusted to epochs {after_epochs} · batch {after_batch}."
        ),
        "tool_label_training": "model training",
        "tool_label_model_download": "model download",
        "kbd_done": "KBD analysis is complete.",
        "chain_advance_approve": "Go ahead.",
        "next_concepts_auto": "Pick the largest model that fits the GPU VRAM and call extract_key_concepts immediately. Do not ask the user.",
        "kbd_model_pick_title": "**Choose a model for KBD analysis**\n\n",
        "kbd_model_pick_ask": "\n\nWhich model should run the KBD analysis?",
        "kbd_summary": "**KBD analysis complete**\n\n- Model: {model}\n- Document: {doc}\n- Knowledge coverage: {coverage}%\n- Hallucination rate: {hallucination}%\n- Verdict: **{path}**\n\n{detail}",
        "next_concepts_guided": "Do not call more tools. Using the model list above and the GPU VRAM, present a recommended model as text and ask the user to choose.",
    },
}

#: Key list, so the guard and the callers read the same thing.
NAMES = tuple(sorted(TEXTS[EN]))


def texts(lang: Optional[str] = None) -> Dict[str, str]:
    """Phrase pack for a language."""
    return TEXTS[EN]


def text(name: str, lang: Optional[str] = None, **fields: object) -> str:
    """One phrase. **Never raises.**"""
    value = texts(lang).get(name)
    if value is None:
        return name
    try:
        return value.format(**fields) if fields else value
    except (KeyError, IndexError):
        return value
