"""locate_model_directory must also resolve the flattened HuggingFace name.

download_hf_task saves the repo id "Qwen/Qwen2.5-3B-Instruct" flattened to
"Qwen--Qwen2.5-3B-Instruct" (safe_name = model_id.replace). The resolver has to
find that copy from the repo id, or the KBD download-then-probe never completes.

pipelines.dialog imports torch, peft and transformers at module level, but
locate_model_directory is pure os logic. A CI venv has none of those heavy
packages, so light stubs are injected before the import (setdefault, so a real
package wins).
"""

import sys
import types
from unittest.mock import patch

import pytest

sys.modules.setdefault("torch", types.ModuleType("torch"))
_peft = types.ModuleType("peft")
_peft.PeftModel = object  # type: ignore[attr-defined]
sys.modules.setdefault("peft", _peft)
_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = object  # type: ignore[attr-defined]
_tf.AutoTokenizer = object  # type: ignore[attr-defined]
_tf.TextIteratorStreamer = object  # type: ignore[attr-defined]
sys.modules.setdefault("transformers", _tf)

from pipelines.dialog import locate_model_directory  # noqa: E402


def _patch_roots(tmp_path, models_dir):
    return (
        patch("pipelines.dialog.BASE_MODELS_ROOT", str(models_dir)),
        patch("pipelines.dialog.TRAINED_OUTPUTS_ROOT", str(tmp_path / "nope_trained")),
        patch("pipelines.dialog.OUTPUTS_ROOT", str(tmp_path / "nope_outputs")),
    )


def test_resolves_flattened_hf_repo_id(tmp_path):
    # The disk holds the flattened name, and a repo id lookup must still find it.
    models = tmp_path / "models"
    (models / "Qwen--Qwen2.5-3B-Instruct").mkdir(parents=True)
    p_base, p_tr, p_out = _patch_roots(tmp_path, models)
    with p_base, p_tr, p_out:
        p = locate_model_directory("Qwen/Qwen2.5-3B-Instruct")
    assert p.endswith("Qwen--Qwen2.5-3B-Instruct")


def test_resolves_plain_name_unchanged(tmp_path):
    # A name without a slash must still resolve (regression guard).
    models = tmp_path / "models"
    (models / "Qwen2.5-0.5B-Instruct").mkdir(parents=True)
    p_base, p_tr, p_out = _patch_roots(tmp_path, models)
    with p_base, p_tr, p_out:
        p = locate_model_directory("Qwen2.5-0.5B-Instruct")
    assert p.endswith("Qwen2.5-0.5B-Instruct")


def test_not_found_raises(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    p_base, p_tr, p_out = _patch_roots(tmp_path, models)
    with p_base, p_tr, p_out:
        with pytest.raises(ValueError):
            locate_model_directory("does/not-exist")
