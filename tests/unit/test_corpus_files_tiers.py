from __future__ import annotations

import itertools
from typing import List, Sequence

from server.core.corpus_files import (
    REFINED_SUFFIXES,
    TIERS,
    _stem,
    count_samples,
    is_data_file,
    is_refined,
    select_data_files,
    tier_of,
)

# ── Three-level precedence ──────────────────────────────────────────────────


def test_twist_beats_preprocessed_and_original():
    assert select_data_files(["qa_dataset.json", "qa_dataset_preprocessed.json", "qa_dataset_twist.json"]) == [
        "qa_dataset_twist.json"
    ]


def test_preprocessed_beats_original_when_no_twist():
    assert select_data_files(["qa_dataset.json", "qa_dataset_preprocessed.json"]) == ["qa_dataset_preprocessed.json"]


def test_legacy_refined_is_the_same_tier_as_preprocessed():
    assert tier_of("qa_dataset_refined.json") == tier_of("qa_dataset_preprocessed.json")
    assert select_data_files(["qa_dataset.json", "qa_dataset_refined.json"]) == ["qa_dataset_refined.json"]


def test_twist_beats_legacy_refined_too():
    assert select_data_files(["qa_dataset_refined.json", "qa_dataset_twist.json"]) == ["qa_dataset_twist.json"]


def test_original_only_folder_is_untouched():
    """Right after an upload or a generation: with no replacement, the original trains."""
    assert select_data_files(["qa_dataset.json"]) == ["qa_dataset.json"]


def test_dlmax_intermediate_never_wins():
    assert select_data_files(["qa_dataset.json", "qa_dataset_preprocessed.json", "qa_dataset_dlmax.json"]) == [
        "qa_dataset_preprocessed.json"
    ]


def test_meta_sidecars_stay_excluded():
    assert select_data_files(["qa_dataset_twist.json", "qa_dataset_refined_stats.json"]) == ["qa_dataset_twist.json"]


def test_empty_folder():
    assert select_data_files([]) == []
    assert select_data_files(["preprocessing_stats.txt", "provenance.json"]) == []


# ── tier_of contract ────────────────────────────────────────────────────────


def test_original_tier_is_below_every_declared_tier():
    """Adding a tier keeps the original at the bottom automatically."""
    assert tier_of("qa_dataset.json") == len(TIERS)
    for index, suffixes in enumerate(TIERS):
        for suffix in suffixes:
            assert tier_of(f"qa_dataset{suffix}.json") == index
            assert tier_of(f"qa_dataset{suffix}.json") < len(TIERS)


def test_refined_suffixes_is_derived_from_tiers():
    """The two declarations cannot diverge — one is derived from the other."""
    assert REFINED_SUFFIXES == tuple(s for tier in TIERS for s in tier)
    assert set(REFINED_SUFFIXES) == {"_twist", "_preprocessed", "_refined"}


def test_is_refined_covers_every_tier():
    for suffix in REFINED_SUFFIXES:
        assert is_refined(f"qa_dataset{suffix}.json") is True
    assert is_refined("qa_dataset.json") is False
    assert is_refined("qa_dataset_refined_stats.json") is False


def _original_is_refined(name: str) -> bool:
    return _stem(name).endswith(("_refined", "_preprocessed"))


def _original_select(filenames: Sequence[str]) -> List[str]:
    data = sorted(n for n in filenames if isinstance(n, str) and is_data_file(n))
    refined = [n for n in data if _original_is_refined(n)]
    if refined:
        data = refined
    json_stems = {_stem(n) for n in data if n.lower().endswith(".json")}
    return [n for n in data if not (n.lower().endswith(".jsonl") and _stem(n) in json_stems)]


#: File names that actually appear in the repository (augmented files aside).
_UNIVERSE_WITHOUT_TWIST = (
    "qa_dataset.json",
    "qa_dataset.jsonl",
    "qa_dataset_preprocessed.json",
    "qa_dataset_refined.json",
    "qa_dataset_refined_stats.json",
    "qa_dataset_dlmax.json",
    "qa_dataset_dlmax.jsonl",
    "preprocessing_stats.txt",
    "generation_params.json",
    "provenance.json",
)


def test_no_regression_on_every_combination_without_twist():
    checked = 0
    for size in range(len(_UNIVERSE_WITHOUT_TWIST) + 1):
        for combo in itertools.combinations(_UNIVERSE_WITHOUT_TWIST, size):
            assert select_data_files(list(combo)) == _original_select(combo), combo
            checked += 1
    assert checked == 2 ** len(_UNIVERSE_WITHOUT_TWIST)


def test_twist_is_the_only_intended_difference():
    """It only differs when an augmented file exists, and always in its favour."""
    folder = ["qa_dataset.json", "qa_dataset_preprocessed.json", "qa_dataset_twist.json"]
    assert _original_select(folder) == ["qa_dataset_preprocessed.json"]
    assert select_data_files(folder) == ["qa_dataset_twist.json"]


# ── Sample counting ─────────────────────────────────────────────────────────
# A dataset arrives as JSON or JSONL. Counting only the JSON shape reports 0
# for every JSONL upload, which silently skips the small-data adjustment.


def test_a_json_list_is_counted(tmp_path):
    f = tmp_path / "qa_dataset.json"
    f.write_text('[{"x": 1}, {"x": 2}, {"x": 3}]', encoding="utf-8")
    assert count_samples(f) == 3


def test_jsonl_is_counted_by_line(tmp_path):
    f = tmp_path / "qa_dataset.jsonl"
    f.write_text('{"x": 1}\n{"x": 2}\n\n{"x": 3}\n', encoding="utf-8")
    assert count_samples(f) == 3


def test_unreadable_or_unexpected_shapes_count_as_zero(tmp_path):
    (tmp_path / "obj.json").write_text('{"not": "a list"}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    assert count_samples(tmp_path / "obj.json") == 0
    assert count_samples(tmp_path / "bad.json") == 0
    assert count_samples(tmp_path / "missing.json") == 0
