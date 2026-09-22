import pytest

from server.core.corpus_target import (
    folder_of,
    is_document,
    scope_documents,
    select_folder,
)

ENTRIES = [
    ("dataset_20260617_041839/.koni_meta/upload.json", 0.0),
    ("dataset_20260617_041839/economics-for-undergraduates.pdf", 0.59),
    ("dataset_20260617_075943/.koni_meta/upload.json", 0.0),
    ("dataset_20260617_075943/llm-based-multi-agent.pdf", 0.52),
    ("dataset_20260902_093459/.koni_meta/upload.json", 0.0),
    ("dataset_20260902_093459/128.pdf", 7.04),
    ("dataset_20260902_093459/197.pdf", 1.79),
]
PICKED = "dataset_20260902_093459"


def test_internal_metadata_is_not_a_document():
    """`.koni_meta` is built so top-level scans do not see it, but rglob is
    recursive and breaks that convention."""
    assert not is_document("ds/.koni_meta/upload.json")
    assert not is_document("ds/.koni_meta/format.json")
    assert is_document("ds/paper.pdf")


@pytest.mark.parametrize("name", ["ds/a.exe", "ds/a.zip", "ds/noext", ""])
def test_non_indexable_files_are_excluded(name):
    assert not is_document(name)


def test_extension_matching_is_case_insensitive():
    assert is_document("ds/PAPER.PDF")


def test_folder_of():
    assert folder_of("ds/a.pdf") == "ds"
    assert folder_of("a.pdf") == ""


# ── Target selection ────────────────────────────────────────────────────────












# ── Does anything downstream undo it? ───────────────────────────────────────




def test_scope_documents_is_a_noop_without_target():
    docs = [{"name": "a/x.pdf"}, {"name": "b/y.pdf"}]
    assert scope_documents(docs, None) == docs


def test_select_folder_with_no_signal():
    assert select_folder(["a", "b"], "", None) is None
