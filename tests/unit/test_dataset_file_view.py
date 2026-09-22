"""corpus/{X}/qa_dataset.json              original
              qa_dataset_preprocessed.json  preprocessed
              qa_dataset_twist.json         augmented    <- what training reads

Deleting the whole folder cannot "undo only the augmentation", so files must be
deletable individually — and for that, every file has to be visible.
"""

from __future__ import annotations

from pathlib import Path

from server.core.corpus_files import PREPROCESSED_FILE, TWIST_FILE, select_data_files
from server.core.dataset_file_view import (
    KIND_DATA,
    KIND_META,
    build_file_entries,
    file_kind,
    remaining_after,
    removable_reason,
    training_ready,
)

_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROUTER = (_ROOT / "server" / "routers" / "data.py").read_text(encoding="utf-8")

_FOLDER = [
    "qa_dataset.json",
    PREPROCESSED_FILE,
    TWIST_FILE,
    "preprocessing_stats.txt",
    "provenance.json",
]


# ── Every file is shown ─────────────────────────────────────────────────────


def test_every_file_is_listed():
    """A hidden file cannot be deleted, and nobody can tell why it is there."""
    names = [e["name"] for e in build_file_entries(_FOLDER)]
    assert names == sorted(_FOLDER)


def test_entries_are_sorted_and_stable():
    assert [e["name"] for e in build_file_entries(list(reversed(_FOLDER)))] == sorted(_FOLDER)


def test_empty_folder():
    assert build_file_entries([]) == []


def test_non_strings_are_ignored():
    assert [e["name"] for e in build_file_entries(["a.json", None, "", 3])] == ["a.json"]  # type: ignore[list-item]


# ── active uses the same decision as the trainer ────────────────────────────


def test_active_marks_exactly_what_trains():
    entries = build_file_entries(_FOLDER)
    active = {e["name"] for e in entries if e["active"]}
    assert active == set(select_data_files(_FOLDER))
    assert active == {TWIST_FILE}


def test_active_follows_the_tier_when_augmentation_is_removed():
    """Deleting the augmented file makes the preprocessed one the training target."""
    remaining = remaining_after(_FOLDER, TWIST_FILE)
    active = {e["name"] for e in build_file_entries(remaining) if e["active"]}
    assert active == {PREPROCESSED_FILE}


def test_active_falls_back_to_the_original():
    remaining = remaining_after(remaining_after(_FOLDER, TWIST_FILE), PREPROCESSED_FILE)
    active = {e["name"] for e in build_file_entries(remaining) if e["active"]}
    assert active == {"qa_dataset.json"}


def test_meta_files_are_never_active():
    for entry in build_file_entries(_FOLDER):
        if entry["kind"] == KIND_META:
            assert entry["active"] is False


def test_kind_separates_data_from_meta():
    assert file_kind("qa_dataset.json") == KIND_DATA
    assert file_kind(TWIST_FILE) == KIND_DATA
    assert file_kind("preprocessing_stats.txt") == KIND_META
    assert file_kind("provenance.json") == KIND_META
    assert file_kind("qa_dataset_refined_stats.json") == KIND_META


def test_sizes_are_carried_when_known():
    entries = {e["name"]: e["size"] for e in build_file_entries(["a.json"], {"a.json": 17})}
    assert entries["a.json"] == 17
    # Unknown means 0 — a missing value is never invented.
    assert build_file_entries(["b.json"])[0]["size"] == 0


# ── Delete decision ─────────────────────────────────────────────────────────


def test_missing_file_is_refused():
    assert removable_reason(_FOLDER, "nope.json") == "not_found"


def test_blank_or_padded_names_are_refused():
    for bad in ("", " qa_dataset.json", "qa_dataset.json "):
        assert removable_reason(_FOLDER, bad) is not None


def test_deleting_the_last_training_file_is_allowed():
    """Not blocked. The opposite of "undo the augmentation" is "undo the

    preprocessing too", which is legitimate. The outcome is reported through
    """
    only = ["qa_dataset.json", "provenance.json"]
    assert removable_reason(only, "qa_dataset.json") is None
    assert training_ready(remaining_after(only, "qa_dataset.json")) is False


def test_training_ready_tracks_the_tier():
    assert training_ready(_FOLDER) is True
    assert training_ready(["provenance.json", "preprocessing_stats.txt"]) is False


# ── Router wiring (source check — a CI venv has no fastapi) ─────────────────


def test_endpoint_exists_and_guards_the_name():
    """Uses the **same guard** as the upload path: `Path(...).name` strips `../`."""
    assert '@router.delete("/datasets/{dataset_id}/files/{file_name}")' in _DATA_ROUTER
    block = _DATA_ROUTER[_DATA_ROUTER.index("async def delete_dataset_file") :][:2200]
    assert "safe_upload_basename(file_name)" in block
    assert "target.resolve().parent != directory.resolve()" in block


def test_endpoint_reports_what_remains():
    block = _DATA_ROUTER[_DATA_ROUTER.index("async def delete_dataset_file") :][:2600]
    assert "training_ready" in block
    assert "build_file_entries" in block
    assert "_invalidate_datasets_cache_for_request(request)" in block


def test_listing_carries_file_entries():
    assert '"file_entries": build_file_entries(' in _DATA_ROUTER
    assert '"files": display_files[:50]' in _DATA_ROUTER


def test_helper_is_import_light():
    import ast

    tree = ast.parse((_ROOT / "server" / "core" / "dataset_file_view.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "typing", "server"}, modules
