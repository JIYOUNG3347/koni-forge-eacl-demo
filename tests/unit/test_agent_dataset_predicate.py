"""Which files in a dataset folder count as QA data.

Metadata written beside the data (generation parameters, preprocessing stats)
is json too, so a plain extension check reads it as QA. ``is_data_file`` is the
single rule, and callers that pick a file must use it.
"""

from __future__ import annotations

from pathlib import Path

from server.core.corpus_files import is_data_file


def test_metadata_json_is_not_picked_as_qa():
    assert is_data_file("generation_params.json") is False
    assert is_data_file("preprocessing_stats.json") is False
    assert is_data_file("qa_dataset.json") is True
    assert is_data_file("qa_dataset.jsonl") is True


def test_non_json_metadata_is_excluded_too():
    for name in ("preprocessing_stats.txt", "generation_params.logs", "generation.log"):
        assert is_data_file(name) is False


def _first_data_file(folder: Path):
    """The callers' fallback: the first data file, no preference order."""
    for f in sorted(folder.iterdir()):
        if f.is_file() and is_data_file(f.name):
            return f
    return None


def test_fallback_skips_metadata_and_finds_the_qa_file(tmp_path):
    (tmp_path / "generation_params.json").write_text("{}")
    (tmp_path / "qa_dataset.json").write_text("[]")
    assert _first_data_file(tmp_path).name == "qa_dataset.json"


def test_fallback_finds_nothing_when_only_metadata_is_present(tmp_path):
    (tmp_path / "generation_params.json").write_text("{}")
    assert _first_data_file(tmp_path) is None
