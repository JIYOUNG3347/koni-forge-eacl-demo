"""Folder uploads (<input webkitdirectory>) send path-prefixed filenames
("subdir/file.pdf") which previously made `dest_dir / filename` write into a
non-existent subdirectory → FileNotFoundError (500). The backend now normalises
to a safe basename via `safe_upload_basename`.

Imported from server.core.helpers (fastapi-free) — NOT from server.routers.data
— so the test collects under the CI unit-test env, which has no fastapi.
"""

from server.core.helpers import safe_upload_basename


def test_strips_folder_prefix():
    assert safe_upload_basename("papers/rehabilitation.pdf") == "rehabilitation.pdf"
    assert safe_upload_basename("subdir/file.pdf") == "file.pdf"
    assert safe_upload_basename("a/b/c/deep.txt") == "deep.txt"


def test_rejects_path_traversal():
    # Directory components (incl. ../) are dropped → only the basename survives.
    assert safe_upload_basename("../escape.pdf") == "escape.pdf"
    assert safe_upload_basename("../../etc/passwd") == "passwd"


def test_plain_filename_unchanged():
    assert safe_upload_basename("report.pdf") == "report.pdf"


def test_empty_or_invalid_returns_none():
    assert safe_upload_basename("") is None
    assert safe_upload_basename(None) is None
    assert safe_upload_basename("   ") is None
    assert safe_upload_basename(".") is None
    assert safe_upload_basename("..") is None
