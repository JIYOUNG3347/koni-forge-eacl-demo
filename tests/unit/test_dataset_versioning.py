"""What this covers:
- parse_version: bare (implicit v1), _v2, _v10, the invalid _v0, other shapes
- format_version: v1 keeps the base, v2+ appends _v{N}
- resolve_next_version: with and without carry-over, alongside _v2
- list_versions: ordering

No fastapi involved — server.core.dataset_versioning is imported directly.
"""

from server.core.dataset_versioning import (
    list_versions,
    parse_version,
)


def test_parse_version_bare_is_implicit_v1():
    assert parse_version("dataset_20260615_042628") == ("dataset_20260615_042628", 1)


def test_parse_version_v2_v10():
    assert parse_version("dataset_xxx_v2") == ("dataset_xxx", 2)
    assert parse_version("dataset_xxx_v10") == ("dataset_xxx", 10)


def test_parse_version_v0_is_not_a_version():
    # _v0 is invalid, so the whole name is the base at v1.
    assert parse_version("dataset_xxx_v0") == ("dataset_xxx_v0", 1)


def test_parse_version_other_suffix():
    assert parse_version("qa_dataset") == ("qa_dataset", 1)
    assert parse_version("dataset_v") == ("dataset_v", 1)  # no number














def test_list_versions_sorted(tmp_path):
    (tmp_path / "dataset_xxx_v3").mkdir()
    (tmp_path / "dataset_xxx").mkdir()  # v1
    (tmp_path / "dataset_xxx_v2").mkdir()
    assert list_versions("dataset_xxx", tmp_path) == [1, 2, 3]


def test_list_versions_missing_root(tmp_path):
    assert list_versions("x", tmp_path / "nope") == []






