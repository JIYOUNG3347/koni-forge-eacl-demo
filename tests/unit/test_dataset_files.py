"""What this covers:
- only source document extensions are listed (.json/.jsonl QA payloads excluded)
- the .koni_meta/ directory is excluded
- sorted by name
- returns name, size and mtime (ISO 8601)
- an empty or missing folder gives an empty list

fastapi-free pure module, so it runs without the web stack.
"""

from datetime import datetime

from server.core.dataset_files import list_dataset_files


def _write(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_lists_source_docs_only(tmp_path):
    folder = tmp_path / "dataset_xxx"
    _write(folder / "b.pdf", "abcde")  # 5 bytes
    _write(folder / "a.docx", "fg")  # 2 bytes
    _write(folder / "qa_dataset.json", "[]")  # QA payload, excluded
    _write(folder / "notes.jsonl", '{"x":1}')  # QA payload, excluded

    files = list_dataset_files(folder)

    names = [f["name"] for f in files]
    assert names == ["a.docx", "b.pdf"]  # sorted by name, json and jsonl excluded
    assert files[0]["size"] == 2
    assert files[1]["size"] == 5
    # The mtime parses as ISO 8601.
    datetime.fromisoformat(files[0]["mtime"])


def test_excludes_koni_meta_dir(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "doc.pdf", "abc")
    _write(folder / ".koni_meta" / "upload.json", "meta-label")

    files = list_dataset_files(folder)

    assert [f["name"] for f in files] == ["doc.pdf"]


def test_supported_extensions(tmp_path):
    folder = tmp_path / "ds"
    for name in ("a.pdf", "b.hwp", "c.hwpx", "d.docx", "e.pptx", "f.txt", "g.md"):
        _write(folder / name, "x")
    _write(folder / "skip.csv", "x")  # unsupported

    files = list_dataset_files(folder)

    assert [f["name"] for f in files] == [
        "a.pdf",
        "b.hwp",
        "c.hwpx",
        "d.docx",
        "e.pptx",
        "f.txt",
        "g.md",
    ]


def test_empty_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert list_dataset_files(folder) == []


def test_missing_folder(tmp_path):
    assert list_dataset_files(tmp_path / "nope") == []


def test_accepts_string_path(tmp_path):
    folder = tmp_path / "ds"
    _write(folder / "a.pdf", "hello")
    files = list_dataset_files(str(folder))
    assert files[0]["name"] == "a.pdf"
    assert files[0]["size"] == 5
