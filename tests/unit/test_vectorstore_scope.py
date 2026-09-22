"""Which collection a VectorStore reads and writes.

Indexing writes one collection per upload folder, so there is no single
collection holding everything. A store built without a name is read-only
across all of them; one built with a name owns that collection.
"""

from __future__ import annotations

import pytest

from modules.rag.vectorstore import VectorStore


class FakeCollection:
    def __init__(self, name, rows):
        self.name = name
        self._rows = rows  # [(doc, distance)]

    def count(self):
        return len(self._rows)

    def query(self, **kwargs):
        n = kwargs.get("n_results", len(self._rows))
        rows = self._rows[:n]
        return {
            "documents": [[d for d, _ in rows]],
            "distances": [[dist for _, dist in rows]],
            "metadatas": [[{"source_folder": self.name} for _ in rows]],
        }


class FakeClient:
    def __init__(self, collections):
        self._collections = collections
        self.deleted = []

    def list_collections(self):
        return self._collections

    def delete_collection(self, name):
        self.deleted.append(name)


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(persist_dir=str(tmp_path))
    vs._client = FakeClient(  # type: ignore[assignment]
        [
            FakeCollection("folder_a", [("chunk from A", 0.4)]),
            FakeCollection("folder_b", [("chunk from B", 0.1)]),
            FakeCollection("empty", []),
        ]
    )
    return vs


class TestUnscopedStore:
    def test_it_claims_no_collection(self, tmp_path):
        assert VectorStore(persist_dir=str(tmp_path)).collection_name is None

    def test_search_spans_every_collection(self, store):
        hits = store.search([0.0], top_k=5)
        assert {h["content"] for h in hits} == {"chunk from A", "chunk from B"}

    def test_hits_are_ordered_by_score_across_collections(self, store):
        hits = store.search([0.0], top_k=5)
        assert [h["content"] for h in hits] == ["chunk from B", "chunk from A"]
        assert hits[0]["score"] == pytest.approx(0.9)

    def test_top_k_applies_to_the_merged_result(self, store):
        assert len(store.search([0.0], top_k=1)) == 1

    def test_an_empty_collection_is_skipped(self, store):
        assert all(h["content"] for h in store.search([0.0], top_k=5))

    def test_writing_without_a_collection_is_refused(self, store):
        """Silently writing to some default collection is how data goes missing."""
        with pytest.raises(RuntimeError, match="collection_name"):
            store.add_document("d1", "d.md", ["x"], [[0.0]])

    def test_deleting_a_dataset_drops_its_collection(self, store):
        store.delete_dataset("folder_a")
        assert store._client.deleted == ["folder_a"]


class TestNamedStore:
    def test_it_keeps_the_name_it_was_given(self, tmp_path):
        assert VectorStore(persist_dir=str(tmp_path), collection_name="folder_a").collection_name == "folder_a"

    def test_listing_chunks_without_a_collection_returns_nothing(self, tmp_path):
        assert VectorStore(persist_dir=str(tmp_path)).list_chunks() == []
