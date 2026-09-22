"""torch- and celery-free pure module, so it runs in a plain CI venv."""

import server.core.corpus_files as cf

REAL = [
    "qa_dataset.json",  # original, 27 records
    "qa_dataset_refined.json",  # refined and augmented, 54 <- what training reads
    "qa_dataset_refined_stats.json",  # metadata (statistics)
    "provenance.json",  # lineage sidecar, not data
    "generation_params.json",  # metadata
]


def test_repro_refined_preferred_over_original():
    # With a refined file present, only that one counts, so the display and training agree at 54.
    assert cf.select_data_files(REAL) == ["qa_dataset_refined.json"]


def test_original_only_when_no_refined():
    # Without a refined file, the original is used.
    assert cf.select_data_files(["qa_dataset.json", "provenance.json"]) == ["qa_dataset.json"]


# ── is_data_file ────────────────────────────────────────────


def test_real_data_files_included():
    assert cf.is_data_file("qa_dataset.json") is True
    assert cf.is_data_file("qa_dataset_refined.json") is True
    assert cf.is_data_file("data.jsonl") is True


def test_non_json_excluded():
    for name in ("README.md", "config.txt", "model.safetensors"):
        assert cf.is_data_file(name) is False


# ── is_refined ──────────────────────────────────────────────
def test_refined_detection():
    assert cf.is_refined("qa_dataset_refined.json") is True
    assert cf.is_refined("qa_dataset.json") is False
    # _refined_stats is metadata and already filtered by is_data_file; its stem is not refined.
    assert cf.is_refined("qa_dataset_refined_stats.json") is False


# ── json / jsonl duplicates ─────────────────────────────────
def test_json_preferred_over_jsonl_same_stem():
    assert cf.select_data_files(["data.json", "data.jsonl"]) == ["data.json"]


def test_different_stem_jsonl_kept():
    got = cf.select_data_files(["a.json", "b.jsonl"])
    assert got == ["a.json", "b.jsonl"]


# ── Defensive cases ─────────────────────────────────────────
def test_empty_and_all_meta():
    assert cf.select_data_files([]) == []
    assert cf.select_data_files(["provenance.json", "generation_params.json"]) == []


def test_non_string_entries_ignored():
    assert cf.select_data_files(["qa_dataset.json", None, 42]) == ["qa_dataset.json"]  # type: ignore[list-item]


def test_multiple_refined_all_kept():
    got = cf.select_data_files(["part1_refined.json", "part2_refined.json", "orig.json"])
    assert got == ["part1_refined.json", "part2_refined.json"]
