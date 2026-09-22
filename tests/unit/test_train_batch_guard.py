"""The batch guard is celery- and torch-free, so it is tested directly."""

import json
from pathlib import Path

import pytest

from server.core.host_config import HOST_CONFIG
from server.core.train_batch_guard import (
    GRAD_CKPT_KEY,
    INFO_KEY,
    PLANNED_KEY,
    apply_batch_guard,
    guard_enabled,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_QWEN3_4B_CFG = {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "intermediate_size": 9728,
}
_A100_FREE_MB = 81000.0


def _model_dir(tmp_path, cfg=_QWEN3_4B_CFG):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _params(model_dir, batch=16, seq=2048, method="lora", accum=4):
    return {
        "model_name_or_path": str(model_dir),
        "mode": method,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": accum,
        "max_seq_length": seq,
    }


# ── flag and skip conditions ─────────────────────────────────────────
def test_guard_enabled_default_on(monkeypatch):
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    assert guard_enabled() is True
    assert guard_enabled({"TRAIN_BATCH_GUARD": "0"}) is False


def test_guard_off_leaves_params_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_BATCH_GUARD", "0")
    p = _params(_model_dir(tmp_path))
    assert apply_batch_guard(p, _A100_FREE_MB) is None
    assert p["per_device_train_batch_size"] == 16 and PLANNED_KEY not in p


def test_planned_marker_skips_second_pass(tmp_path, monkeypatch):
    """Params already planned are left alone, so the plan cannot drift."""
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    p = _params(_model_dir(tmp_path))
    p[PLANNED_KEY] = True
    assert apply_batch_guard(p, _A100_FREE_MB) is None
    assert p["per_device_train_batch_size"] == 16


def test_unknown_free_or_shape_never_blocks(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    p = _params(_model_dir(tmp_path))
    assert apply_batch_guard(p, None) is None  # free memory unknown
    no_cfg = tmp_path / "empty"
    no_cfg.mkdir()
    assert apply_batch_guard(_params(no_cfg), _A100_FREE_MB) is None  # no config.json


def test_incident_case_clamps_and_preserves_effective(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    p = _params(_model_dir(tmp_path), batch=16, seq=2048, method="lora", accum=4)
    info = apply_batch_guard(p, _A100_FREE_MB, where="test")
    assert info is not None and info["adjusted"] is True
    assert 1 <= p["per_device_train_batch_size"] < 16
    assert p["per_device_train_batch_size"] * p["gradient_accumulation_steps"] >= 64
    assert p["batch_size"] == p["per_device_train_batch_size"]  # alias stays in sync
    assert p[PLANNED_KEY] is True and p[INFO_KEY] == info


def test_small_request_passes_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    p = _params(_model_dir(tmp_path), batch=2, seq=512, accum=4)
    info = apply_batch_guard(p, _A100_FREE_MB)
    assert info is not None and info["adjusted"] is False
    assert p["per_device_train_batch_size"] == 2 and p["gradient_accumulation_steps"] == 4
    assert GRAD_CKPT_KEY not in p


def test_model_limits_cap_applied(tmp_path, monkeypatch):
    """A request above the host.toml ceiling is capped."""
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    cap = HOST_CONFIG.model_limits.max_train_batch_size
    p = _params(_model_dir(tmp_path), batch=cap * 4, seq=512)
    info = apply_batch_guard(p, _A100_FREE_MB * 100)  # plenty of memory — only the cap applies
    assert info is not None
    assert p["per_device_train_batch_size"] <= cap
    assert any("max_train_batch_size" in r for r in info["reasons"])


def test_impossible_raises_clear_error(tmp_path, monkeypatch):
    """Not even batch 1 with checkpointing fits — refuse instead of OOM."""
    monkeypatch.delenv("TRAIN_BATCH_GUARD", raising=False)
    p = _params(_model_dir(tmp_path), batch=4, method="sft")  # 4B full FT is ~64GB static
    with pytest.raises(RuntimeError, match="batch_guard"):
        apply_batch_guard(p, 8_000.0)  # 8GB free


def test_training_task_applies_the_guard():
    """Static check: the worker must call the guard before loading weights."""
    src = (PROJECT_ROOT / "celery_app/tasks/train_tasks.py").read_text(encoding="utf-8")
    body = src.split("def _run_sft_training(")[1]
    assert "apply_batch_guard(" in body
    assert body.index("apply_batch_guard(") < body.index("AutoModelForCausalLM.from_pretrained(")


# ── shared safe batch used by the recommendation path ────────────────
def test_safe_batch_for_model_returns_bound(tmp_path, monkeypatch):
    from server.core.train_batch_guard import safe_batch_for_model

    model = _model_dir(tmp_path)
    bs = safe_batch_for_model(model, "lora", 2048, _A100_FREE_MB)
    assert bs is not None and 1 <= bs < 16


def test_safe_batch_for_model_unknown_is_none(tmp_path):
    from server.core.train_batch_guard import safe_batch_for_model

    empty = tmp_path / "no_model"
    empty.mkdir()
    assert safe_batch_for_model(empty, "lora", 2048, _A100_FREE_MB) is None  # no config.json
    assert safe_batch_for_model(_model_dir(tmp_path), "lora", 2048, 0) is None  # free memory unknown
