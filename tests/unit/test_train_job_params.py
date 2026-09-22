"""[TOOL-EXEC] tuning: 'start_training_job' EXCEPTION: 'job_name'
    Tool loop detected ('start_training_job' x2) — breaking loop

The tool schema declares required fields, but a local model does not honour
them. The implementation indexed params["job_name"] directly and raised
KeyError, and all the LLM got back was the string 'job_name'.
"""

from pathlib import Path

from server.core.train_job_params import (
    derive_job_name,
    missing_params_message,
    missing_required,
    resolve_training_params,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Recovery: the code fills in what it can derive ───────────────────
def test_job_name_derived_from_dataset_when_missing():
    out = resolve_training_params({"base_model": "Qwen--Qwen3-4B", "dataset": "agent-gen-123_twist"})
    assert out["ok"] is True
    assert out["params"]["job_name"] == "agent-gen-123_twist_training"


def test_explicit_job_name_is_kept():
    out = resolve_training_params({"job_name": "my_run", "base_model": "m", "dataset": "d"})
    assert out["params"]["job_name"] == "my_run"


def test_dataset_falls_back_to_pipeline_context():
    # A missing dataset in the tool input is recovered from what the pipeline knows.
    out = resolve_training_params({"base_model": "m"}, fallback_dataset="ds_twist")
    assert out["ok"] is True
    assert out["params"]["dataset"] == "ds_twist"
    assert out["params"]["job_name"] == "ds_twist_training"


def test_derive_does_not_double_suffix():
    assert derive_job_name("ds_training") == "ds_training"
    assert derive_job_name("ds") == "ds_training"
    assert derive_job_name("  ds  ") == "ds_training"
    assert derive_job_name("") == ""


# ── Honest failure: an unrecoverable gap is stated so it can be fixed ─


def test_blank_strings_count_as_missing():
    out = resolve_training_params({"base_model": "  ", "dataset": ""})
    assert out["ok"] is False
    assert missing_required({"base_model": " ", "dataset": "d"}) == ["base_model"]


def test_message_lists_only_missing_fields():
    assert "dataset" in missing_params_message(["dataset"])
    assert "base_model" not in missing_params_message(["dataset"])


def test_original_params_are_not_mutated():
    src = {"base_model": "m", "dataset": "d"}
    resolve_training_params(src)
    assert "job_name" not in src  # the input dict is left alone


def test_tuning_specialist_no_longer_raises_keyerror():
    src = (PROJECT_ROOT / "modules/agents/specialists/tuning_specialist.py").read_text(encoding="utf-8")
    assert "resolve_training_params(" in src
    # Check the direct access that raised KeyError now comes after recovery.
    idx_resolve = src.index("resolve_training_params(")
    idx_direct = src.index('job_name = params["job_name"]')
    assert idx_resolve < idx_direct, "recovery must come before the direct access"
