"""Why promotion existed. With `generated_corpus` (in progress) and `corpus`
(trainable) — "location is state" — preprocessed data had to be **moved** into
the training root. That promotion had split into four implementations that
behaved differently, and all of them are now gone.
"""

from __future__ import annotations

import ast
from pathlib import Path

from server.core.corpus_files import PREPROCESSED_FILE, TWIST_FILE, select_data_files

_ROOT = Path(__file__).resolve().parents[2]

_DATA_ROUTER = (_ROOT / "server" / "routers" / "data.py").read_text(encoding="utf-8")
_REFINE = (_ROOT / "celery_app" / "tasks" / "data_tasks.py").read_text(encoding="utf-8")


def _code(source: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


# ── The promotion machinery is entirely gone ────────────────────────────────


def test_promotion_module_is_gone():
    assert not (_ROOT / "server" / "core" / "dataset_ready.py").exists()
    for name in ("dataset_ready", "split_promotable", "has_augmented_counterpart"):
        assert name not in _code(_DATA_ROUTER), name


def test_confirm_endpoint_is_gone():
    assert "/confirm" not in _code(_DATA_ROUTER)
    assert "confirm_dataset" not in _code(_DATA_ROUTER)
    assert "ConfirmRequest" not in _code(_DATA_ROUTER)


def test_refine_no_longer_copies_to_another_root():
    """refine moves its result nowhere — it is already in the training root."""
    code = _code(_REFINE)
    assert "copytree" not in code
    assert "promoted_to_training" not in code
    assert "promotion_error" not in code


def test_the_second_bucket_is_gone():
    """There is no second root to scan, and no response key that carried it."""
    code = _code(_DATA_ROUTER)
    assert 'scan_dir(GENERATED_DIR, "processed")' not in code
    assert "GENERATED_DIR" not in code
    assert '"processed": processed' not in code
    block = code[code.index("response = {") :][:260]
    assert '"raw": raw' in block and '"training": training' in block
    assert '"processed"' not in block


# ── The decision shares a source with the trainer ───────────────────────────


def test_record_count_uses_the_trainer_rule():
    assert "_selected = select_data_files(file_names)" in _code(_DATA_ROUTER)


# ── The precedence rule takes its place ─────────────────────────────────────


def test_tier_decides_what_trains_now(tmp_path: Path):
    """What promotion answered — "is this folder ready to train?" — the file name now answers."""
    assert select_data_files(["qa_dataset.json"]) == ["qa_dataset.json"]
    assert select_data_files(["qa_dataset.json", PREPROCESSED_FILE]) == [PREPROCESSED_FILE]
    assert select_data_files(["qa_dataset.json", PREPROCESSED_FILE, TWIST_FILE]) == [TWIST_FILE]


def test_training_endpoint_reads_the_single_root():
    """Pin the premise itself: training starts without promotion."""
    from server.core.dataset_paths import corpus_root, dataset_roots

    src = (_ROOT / "server" / "routers" / "train.py").read_text(encoding="utf-8")
    assert "dataset_roots()" in src
    assert dataset_roots() == (corpus_root(),)


def test_router_has_no_syntax_damage_from_the_removal():
    """This PR removed endpoints line by line, so the syntax itself is checked."""
    ast.parse(_DATA_ROUTER)


def test_in_flight_generation_folders_stay_out_of_the_list():
    """A folder still being generated (only `generation_params.json`) must not be listed."""
    code = _code(_DATA_ROUTER)
    assert 'if category == "processed" and not has_output:' not in code
    assert 'if category != "raw" and not has_output:' in code
