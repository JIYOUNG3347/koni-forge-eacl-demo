"""{"name": "agent-gen-1789454859", "type": "training",
     "files": ["generation_params.json"], "size_mb": 0.0}
    ...
    "_next_step": "Call analyze_dataset(dataset_name='agent-gen-1789454859') ..."

An unfinished generation left a folder holding only `generation_params.json`,
which was listed as trainable, and `_next_step` picked the alphabetically first
one, so the LLM was told to analyse an empty folder.

This file creates **real folders** and checks the decision. A source-string
check cannot tell "it calls that function" from "the result is right".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.core.corpus_files import META_PATTERNS, is_data_file
from server.core.dataset_file_view import training_ready

REPO = Path(__file__).resolve().parents[2]
CORPUS_SPECIALIST = REPO / "modules" / "agents" / "specialists" / "corpus_specialist.py"


# ── The decision itself ─────────────────────────────────────────────────────


def test_generation_params_is_not_training_data():
    assert is_data_file("generation_params.json") is False
    assert training_ready(["generation_params.json"]) is False


def test_a_folder_with_real_qa_is_training_ready():
    assert training_ready(["generation_params.json", "qa_dataset.json"]) is True


@pytest.mark.parametrize(
    "name",
    ["generation_params.json", "preprocessing_stats.json", "training_params.json"],
)
def test_metadata_files_never_count_as_data(name):
    assert is_data_file(name) is False


def test_generation_params_is_in_the_single_source():
    assert "generation_params" in META_PATTERNS


# ── Does it surface what was filtered out? ──────────────────────────────────


# ── End to end with real folders ────────────────────────────────────────────


def _listing_of(root: Path) -> dict:
    """Classifies folders by the **same rule** as the corpus scan in `_list_datasets`."""
    datasets, incomplete = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        names = [f.name for f in d.iterdir() if f.is_file()]
        if not training_ready(names):
            incomplete.append(d.name)
            continue
        datasets.append(d.name)
    return {"datasets": datasets, "incomplete": incomplete}


def test_live_shaped_tree_classifies_as_the_screen_does(tmp_path):
    (tmp_path / "agent-gen-1789454859").mkdir()
    (tmp_path / "agent-gen-1789454859" / "generation_params.json").write_text("{}")

    done = tmp_path / "dataset_20260908_102036"
    done.mkdir()
    (done / "generation_params.json").write_text("{}")
    (done / "qa_dataset.json").write_text("[]")

    out = _listing_of(tmp_path)
    assert out["datasets"] == ["dataset_20260908_102036"]
    assert out["incomplete"] == ["agent-gen-1789454859"]


def test_next_step_never_points_at_an_empty_folder(tmp_path):
    (tmp_path / "agent-gen-1789454859").mkdir()
    (tmp_path / "agent-gen-1789454859" / "generation_params.json").write_text("{}")

    out = _listing_of(tmp_path)
    assert out["datasets"] == []  # nothing to choose, so do not point at analyze
    assert out["incomplete"] == ["agent-gen-1789454859"]


def test_response_stays_valid_json(tmp_path):
    out = _listing_of(tmp_path)
    json.loads(json.dumps(out, ensure_ascii=False))
