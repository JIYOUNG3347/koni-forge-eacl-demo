"""Pure module — no fastapi/celery. Covers write/read round-trip, missing-file
handling (pre-v2 datasets), source-file collection, and backward-compatible
deserialization.
"""

from pathlib import Path

from server.core.provenance import (
    PROVENANCE_FILENAME,
    SCHEMA_VERSION,
    ProvenanceRecord,
    collect_source_files,
    read_provenance,
    write_upload_label,
)


def test_read_missing_returns_none(tmp_path: Path):
    """Pre-v2 datasets have no sidecar → None (unavailable), not an error."""
    assert read_provenance(tmp_path) is None
    assert read_provenance(tmp_path / "does_not_exist") is None


def test_read_corrupt_returns_none(tmp_path: Path):
    (tmp_path / PROVENANCE_FILENAME).write_text("{not valid json", encoding="utf-8")
    assert read_provenance(tmp_path) is None


def test_read_non_dict_returns_none(tmp_path: Path):
    (tmp_path / PROVENANCE_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    assert read_provenance(tmp_path) is None


def test_collect_source_files(tmp_path: Path):
    (tmp_path / "b.pdf").write_text("xx", encoding="utf-8")
    (tmp_path / "a.pdf").write_text("y", encoding="utf-8")
    (tmp_path / "sub").mkdir()  # directories are ignored
    files = collect_source_files(tmp_path)
    assert [f.name for f in files] == ["a.pdf", "b.pdf"]  # sorted
    assert files[0].size == 1 and files[1].size == 2
    assert all(f.mtime for f in files)


def test_collect_source_files_skip(tmp_path: Path):
    (tmp_path / "a.pdf").write_text("y", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    files = collect_source_files(tmp_path, skip_names={"provenance.json"})
    assert [f.name for f in files] == ["a.pdf"]


def test_collect_source_files_missing_dir(tmp_path: Path):
    assert collect_source_files(tmp_path / "nope") == []




def test_from_dict_backward_compat_bare_filenames():
    """Tolerate a legacy source_files list of bare strings."""
    rec = ProvenanceRecord.from_dict({"source_folder": "ds", "source_files": ["x.pdf", "y.pdf"]})
    assert [f.name for f in rec.source_files] == ["x.pdf", "y.pdf"]
    assert rec.source_files[0].size == 0
    assert rec.schema_version == SCHEMA_VERSION


def test_from_dict_defaults_empty():
    rec = ProvenanceRecord.from_dict({})
    assert rec.source_folder == ""
    assert rec.source_files == []
    assert rec.generator == {}
    assert rec.pipeline_steps == []
    assert rec.source_folder_label is None










def test_upload_label_stored_in_subdir_invisible_to_file_scan(tmp_path: Path):
    """Label sidecar must live in a subdirectory so top-level is_file() scans
    (RAG indexer / generator / source_files) never pick it up."""
    (tmp_path / "doc.pdf").write_text("x", encoding="utf-8")
    write_upload_label(tmp_path, "folderA")
    top_level_files = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert top_level_files == ["doc.pdf"]  # .koni_meta is a dir, excluded
    assert [f.name for f in collect_source_files(tmp_path)] == ["doc.pdf"]


