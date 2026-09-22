"""What this covers:
- folder_path, disk_size_bytes, file_count and last_modified are correct
- the .koni_meta/ subdirectory is excluded from both size and count
- metadata sidecars (provenance.json) are excluded from file_count but counted in size
- empty and missing folders
- only supported extensions count towards file_count

fastapi-free pure module, so it runs without the web stack.
"""

import os
from datetime import datetime

from server.core.storage_info import compute_storage_info


def _write(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_basic_counts_and_size(tmp_path):
    folder = tmp_path / "dataset_20260615_000000"
    _write(folder / "a.pdf", "abcde")  # 5 bytes, supported
    _write(folder / "b.docx", "fghij")  # 5 bytes, supported
    _write(folder / "qa_dataset.json", "[]")  # 2 bytes, supported

    info = compute_storage_info(folder)

    assert info["folder_path"] == str(folder)
    assert info["file_count"] == 3
    assert info["disk_size_bytes"] == 12
    # It must parse as ISO 8601.
    assert info["last_modified"] is not None
    datetime.fromisoformat(info["last_modified"])


def test_excludes_koni_meta_dir_from_size_and_count(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "doc.pdf", "abcde")  # 5 bytes
    # .koni_meta/upload.json is excluded from both size and count.
    _write(folder / ".koni_meta" / "upload.json", "this-is-meta-payload")

    info = compute_storage_info(folder)

    assert info["file_count"] == 1
    assert info["disk_size_bytes"] == 5  # .koni_meta bytes not included


def test_meta_sidecar_excluded_from_count_but_in_size(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "doc.pdf", "abc")  # 3 bytes, counted
    _write(folder / "provenance.json", "meta!")  # 5 bytes, NOT counted
    _write(folder / "twist.log", "loglog")  # 6 bytes, NOT counted (.log)

    info = compute_storage_info(folder)

    assert info["file_count"] == 1  # only doc.pdf
    assert info["disk_size_bytes"] == 14  # 3 + 5 + 6 (footprint includes meta)


def test_only_supported_extensions_counted(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "a.pdf", "ab")  # supported → counted
    _write(folder / "b.xyz", "cd")  # unsupported → not counted (but in size)

    info = compute_storage_info(folder)

    assert info["file_count"] == 1
    assert info["disk_size_bytes"] == 4


def test_empty_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()

    info = compute_storage_info(folder)

    assert info["file_count"] == 0
    assert info["disk_size_bytes"] == 0
    assert info["last_modified"] is None
    assert info["folder_path"] == str(folder)


def test_missing_folder_returns_zeroes(tmp_path):
    folder = tmp_path / "does_not_exist"

    info = compute_storage_info(folder)

    assert info == {
        "folder_path": str(folder),
        "disk_size_bytes": 0,
        "file_count": 0,
        "last_modified": None,
    }


def test_last_modified_is_latest(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "old.pdf", "a")
    _write(folder / "new.pdf", "b")
    old_t = datetime(2020, 1, 1).timestamp()
    new_t = datetime(2026, 6, 15, 2, 35).timestamp()
    os.utime(folder / "old.pdf", (old_t, old_t))
    os.utime(folder / "new.pdf", (new_t, new_t))

    info = compute_storage_info(folder)

    assert info["last_modified"] == datetime.fromtimestamp(new_t).isoformat()


def test_accepts_string_path(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "a.txt", "hello")

    info = compute_storage_info(str(folder))

    assert info["file_count"] == 1
    assert info["disk_size_bytes"] == 5
