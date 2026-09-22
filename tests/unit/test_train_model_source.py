"""Resolving the training base model.
- a base model name resolves under {storage}/models/{name}
- a model_path (completed output or checkpoint) is accepted
- a path outside the allowed roots raises ValueError (confinement)
- '..' traversal in either argument raises ValueError
- a missing path raises ValueError
- model_path wins over model_name

Pure module, free of fastapi and torch, so it runs in a plain CI venv.
"""

import pytest

from server.core.train_model_source import resolve_train_model_source


def _mk(storage, *parts):
    d = storage.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_base_model_by_name(tmp_path):
    _mk(tmp_path, "models", "gemma-3-1b")
    out = resolve_train_model_source("gemma-3-1b", None, tmp_path, "u1")
    assert out == str((tmp_path / "models" / "gemma-3-1b").resolve())


def test_completed_model_by_path(tmp_path):
    p = _mk(tmp_path, "outputs", "u1", "completed", "my-sft-job")
    out = resolve_train_model_source("my-sft-job", str(p), tmp_path, "u1")
    assert out == str(p.resolve())


def test_checkpoint_model_by_path(tmp_path):
    p = _mk(tmp_path, "checkpoints", "some-job", "round_1")
    out = resolve_train_model_source("x", str(p), tmp_path, "u1")
    assert out == str(p.resolve())


def test_model_path_takes_precedence(tmp_path):
    _mk(tmp_path, "models", "base-a")
    p = _mk(tmp_path, "outputs", "u1", "completed", "trained-b")
    out = resolve_train_model_source("base-a", str(p), tmp_path, "u1")
    assert out == str(p.resolve())  # model_path wins


def test_path_outside_allowed_root_rejected(tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    with pytest.raises(ValueError, match="Model path is not allowed"):
        resolve_train_model_source("x", str(outside), tmp_path, "u1")


def test_path_traversal_rejected(tmp_path):
    _mk(tmp_path, "models", "base-a")
    # Escaping outputs/u1/completed with '..' lands outside the allowed roots.
    evil = tmp_path / "outputs" / "u1" / "completed" / ".." / ".." / ".." / "etc"
    (tmp_path / "etc").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Model path is not allowed"):
        resolve_train_model_source("x", str(evil), tmp_path, "u1")


def test_other_user_completed_rejected(tmp_path):
    # u1 reaching for u2's completed model is outside u1's allowed roots.
    p = _mk(tmp_path, "outputs", "u2", "completed", "u2-job")
    with pytest.raises(ValueError, match="Model path is not allowed"):
        resolve_train_model_source("x", str(p), tmp_path, "u1")


def test_nonexistent_path_rejected(tmp_path):
    missing = tmp_path / "outputs" / "u1" / "completed" / "nope"
    with pytest.raises(ValueError, match="does not exist"):
        resolve_train_model_source("x", str(missing), tmp_path, "u1")


def test_nonexistent_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        resolve_train_model_source("ghost", None, tmp_path, "u1")


def test_name_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="Model name is not allowed"):
        resolve_train_model_source("../../etc", None, tmp_path, "u1")


def test_empty_both_rejected(tmp_path):
    with pytest.raises(ValueError, match="is required"):
        resolve_train_model_source("", None, tmp_path, "u1")
