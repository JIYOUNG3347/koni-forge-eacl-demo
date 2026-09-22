"""
VectorStore — ChromaDB-backed vector store

Handles document indexing, search and deletion.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag.vectorstore")


def _default_chroma_dir():
    """Chroma path. ``server.core.config`` exports CHROMA_PERSIST_DIR at import,
    so the fallback only applies when this module is used on its own."""
    for env_key in ("CHROMA_PERSIST_DIR", "CHROMA_DIR"):
        val = os.getenv(env_key)
        if val:
            return val
    storage = os.getenv("STORAGE_BASE_PATH")
    if storage:
        return str(Path(storage) / "chroma")
    return str(Path(__file__).resolve().parents[2] / "storage" / "chroma")


DEFAULT_CHROMA_DIR = _default_chroma_dir()

# Chunking defaults.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into chunks.
    Prefers paragraph and sentence boundaries, falling back to character count.
    """
    if not text.strip():
        return []

    # Split on paragraphs first.
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # A paragraph longer than chunk_size splits on sentences. The
            # ideographic full stop is included so CJK source documents split too.
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?。])\s+", para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= chunk_size:
                        sub = (sub + " " + sent).strip() if sub else sent
                    else:
                        if sub:
                            chunks.append(sub)
                        # A sentence that is still too long splits on character count.
                        if len(sent) > chunk_size:
                            for i in range(0, len(sent), chunk_size - chunk_overlap):
                                chunks.append(sent[i : i + chunk_size])
                            sub = ""
                        else:
                            sub = sent
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


class VectorStore:
    """ChromaDB vector store."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """``collection_name`` is required to write. Without one the store is
        read-only across every collection, which is what an unscoped search
        needs: indexing writes one collection per upload folder, so there is no
        single collection holding everything."""
        self.persist_dir = persist_dir or os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._doc_index_path = Path(self.persist_dir) / "document_index.json"

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("The chromadb package is required. Run 'pip install chromadb'.")

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        # Only a named store owns a collection. Creating one by default left an
        # empty collection behind that every unscoped search then queried.
        if self.collection_name:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(f"ChromaDB initialized: {self.persist_dir}, collection={self.collection_name or '(all)'}")

    def _require_collection(self, action: str):
        """Guard for the operations that need one named collection."""
        self._ensure_client()
        if self._collection is None:
            raise RuntimeError(f"{action} needs a collection_name: VectorStore(collection_name=...)")

    def _load_doc_index(self) -> List[Dict]:
        if self._doc_index_path.exists():
            try:
                return json.loads(self._doc_index_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_doc_index(self, index: List[Dict]):
        self._doc_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._doc_index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_document(
        self,
        doc_id: str,
        doc_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict] = None,
        page_nums: Optional[List[Optional[int]]] = None,
    ) -> int:
        """
        Add a document's chunks and embeddings to the store.
        A document of the same name already indexed elsewhere is removed first.
        Returns: number of chunks added.
        """
        self._require_collection("add_document")

        # ── Remove a duplicate document ──
        # The same doc_name indexed under any dataset is dropped first.
        doc_index = self._load_doc_index()
        duplicates = [d for d in doc_index if d.get("name") == doc_name and d.get("id") != doc_id]
        for dup in duplicates:
            dup_id = dup.get("id", "")
            try:
                self._collection.delete(where={"doc_id": dup_id})
                logger.info(
                    f"Removed duplicate index for '{doc_name}' (old id={dup_id}, dataset={dup.get('source_folder', '?')})"
                )
            except Exception as e:
                logger.warning(f"Failed to remove duplicate chunks for '{dup_id}': {e}")
        if duplicates:
            doc_index = [d for d in doc_index if not (d.get("name") == doc_name and d.get("id") != doc_id)]

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        def _chunk_meta(i: int) -> Dict[str, Any]:
            meta: Dict[str, Any] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_index": i,
                **(metadata or {}),
            }
            if page_nums is not None and i < len(page_nums) and page_nums[i] is not None:
                meta["page_num"] = page_nums[i]
            return meta

        metadatas = [_chunk_meta(i) for i in range(len(chunks))]

        # ChromaDB batch limit
        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self._collection.add(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        # Update document index
        doc_index = [d for d in doc_index if d.get("id") != doc_id]
        doc_index.append(
            {
                "id": doc_id,
                "name": doc_name,
                "chunks": len(chunks),
                "status": "indexed",
                "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                **(metadata or {}),
            }
        )
        self._save_doc_index(doc_index)

        logger.info(f"Indexed document '{doc_name}': {len(chunks)} chunks")
        return len(chunks)

    def list_chunks(self) -> List[Dict[str, Any]]:
        """Read-only. A missing collection returns an empty list rather than
        creating one. Older chunks without a page_num key surface as None.

        """
        try:
            import chromadb
        except ImportError:
            return []
        if not self.collection_name:
            return []
        try:
            client = self._client or chromadb.PersistentClient(path=self.persist_dir)
            col = client.get_collection(name=self.collection_name)
            res = col.get(include=["documents", "metadatas"])
        except Exception:
            return []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        records: List[Dict[str, Any]] = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            records.append(
                {
                    "chunk_index": meta.get("chunk_index"),
                    "page_num": meta.get("page_num"),
                    "doc_name": meta.get("doc_name"),
                    "text": doc or "",
                }
            )
        return records

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
Search for similar documents using the query embedding.

        With a collection_name, that collection is searched. Without one, every
        collection is searched and the hits are merged by score — indexing
        writes one collection per upload folder, so this is what "search
        everything" means.

        Returns: [{"content", "score", "metadata"}, ...]
        """
        self._ensure_client()

        if self._collection is None:
            return self._search_all(query_embedding, top_k, where)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                # ChromaDB cosine distance → similarity
                similarity = 1.0 - distance
                output.append(
                    {
                        "content": doc,
                        "score": round(similarity, 4),
                        "raw_score": round(similarity, 4),
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    }
                )

        return output

    def _search_all(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search every collection and keep the best hits overall."""
        hits: List[Dict[str, Any]] = []
        for col in self._client.list_collections():
            try:
                if col.count() == 0:
                    continue
                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, col.count()),
                    "include": ["documents", "distances", "metadatas"],
                }
                if where:
                    kwargs["where"] = where
                hits.extend(self._rows(col.query(**kwargs)))
            except Exception as e:  # noqa: BLE001 — one bad collection must not end the search
                logger.warning(f"Search skipped collection '{col.name}': {e}")
        hits.sort(key=lambda r: r["score"], reverse=True)
        return hits[:top_k]

    @staticmethod
    def _rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ChromaDB query output to result rows. Cosine distance to similarity."""
        output: List[Dict[str, Any]] = []
        if not results.get("documents") or not results["documents"][0]:
            return output
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 0
            similarity = 1 - distance
            output.append(
                {
                    "content": doc,
                    "score": round(similarity, 4),
                    "raw_score": round(similarity, 4),
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
            )
        return output

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        self._require_collection("delete_document")

        try:
            self._collection.delete(where={"doc_id": doc_id})

            doc_index = self._load_doc_index()
            doc_index = [d for d in doc_index if d.get("id") != doc_id]
            self._save_doc_index(doc_index)

            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Vector store status."""
        try:
            self._ensure_client()
            count = (
                self._collection.count()
                if self._collection is not None
                else sum(c.count() for c in self._client.list_collections())
            )
            doc_index = self._load_doc_index()

            total_size = sum(f.stat().st_size for f in Path(self.persist_dir).rglob("*") if f.is_file())

            return {
                "status": "active",
                "backend": "ChromaDB",
                "persist_dir": self.persist_dir,
                "collection": self.collection_name or "(all)",
                "total_chunks": count,
                "total_documents": len(doc_index),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "documents": doc_index,
            }
        except Exception as e:
            return {
                "status": "not_initialized",
                "error": str(e),
                "persist_dir": self.persist_dir,
            }

    def list_documents(self) -> List[Dict]:
        """Indexed documents."""
        return self._load_doc_index()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Documents grouped by dataset (source_folder)."""
        doc_index = self._load_doc_index()
        datasets: Dict[str, Dict[str, Any]] = {}

        for doc in doc_index:
            folder = doc.get("source_folder", doc.get("name", "unknown"))
            if folder not in datasets:
                datasets[folder] = {
                    "dataset": folder,
                    "total_chunks": 0,
                    "documents": [],
                    "created_at": doc.get("created_at", ""),
                }
            datasets[folder]["total_chunks"] += doc.get("chunks", 0)
            datasets[folder]["documents"].append(doc)
            # Keep the most recent created_at.
            doc_ts = doc.get("created_at", "")
            if doc_ts > datasets[folder]["created_at"]:
                datasets[folder]["created_at"] = doc_ts

        return sorted(datasets.values(), key=lambda d: d["created_at"], reverse=True)

    def delete_dataset(self, dataset_name: str) -> int:
        """Delete a dataset: its collection and its entries in the index.

        Indexing writes one collection per upload folder, so the collection is
        the dataset. Removing its documents one by one from some other
        collection leaves the data in place.

        Returns the number of documents removed from the index.
        """
        self._ensure_client()
        try:
            self._client.delete_collection(name=dataset_name)
        except Exception as e:  # noqa: BLE001 — a missing collection is normal cleanup
            logger.info(f"Collection '{dataset_name}' not dropped: {e}")

        doc_index = self._load_doc_index()
        kept = [d for d in doc_index if d.get("source_folder") != dataset_name]
        deleted = len(doc_index) - len(kept)
        if deleted:
            self._save_doc_index(kept)

        logger.info(f"Deleted dataset '{dataset_name}': {deleted} documents")
        return deleted

    def search_by_dataset(
        self,
        query_embedding: List[float],
        dataset_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search within one dataset only.

        When dataset_name matches a ChromaDB collection name, that collection is
        opened directly (indexing stores collection name = dataset name).
        Otherwise fall back to the default collection with a source_folder filter.
        """
        self._ensure_client()
        try:
            col = self._client.get_collection(name=dataset_name)
            count = col.count()
            if count == 0:
                logger.warning(f"Collection '{dataset_name}' is empty, falling back to metadata filter")
                raise ValueError("empty collection")
            n = min(top_k, count)
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "distances", "metadatas"],
            )
            output = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1.0 - distance
                    output.append(
                        {
                            "content": doc,
                            "score": round(similarity, 4),
                        "raw_score": round(similarity, 4),  # same as search()
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        }
                    )
            return output
        except Exception as exc:
            logger.debug(f"search_by_dataset direct-collection failed ({exc}), using metadata filter")
            return self.search(
                query_embedding,
                top_k=top_k,
                where={"source_folder": dataset_name},
            )
