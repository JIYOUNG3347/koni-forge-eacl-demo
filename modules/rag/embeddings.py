"""EmbeddingManager — loads sentence-transformers models and builds embeddings."""

import logging
import os
from typing import List, Optional

logger = logging.getLogger("rag.embeddings")

# Supported embedding models.
SUPPORTED_MODELS = {
    "intfloat/multilingual-e5-large": {
        "dim": 1024,
        "description": "Multilingual E5 Large — strong multilingual quality",
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
    },
    "intfloat/multilingual-e5-base": {
        "dim": 768,
        "description": "Multilingual E5 Base — lighter than E5 Large",
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
    },
    "jhgan/ko-sroberta-multitask": {
        "dim": 768,
        "description": "Korean-specialised SRoBERTa — best for Korean-only tasks",
        "prefix_query": "",
        "prefix_passage": "",
    },
    "BAAI/bge-m3": {
        "dim": 1024,
        "description": "BGE-M3 — multilingual, dense/sparse/colbert multi-vector",
        "prefix_query": "",
        "prefix_passage": "",
    },
}

DEFAULT_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "multilingual-e5-large")

# Short alias to full HuggingFace model id, so an alias from the API or .env resolves.
ALIAS_MAP = {
    "multilingual-e5": "intfloat/multilingual-e5-base",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
    "bge-m3": "BAAI/bge-m3",
}


class EmbeddingManager:
    """Loads an embedding model and produces embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        raw_name = model_name or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)
        # Normalise an alias to the full HF id (a full id passes through).
        self.model_name = ALIAS_MAP.get(raw_name, raw_name)
        self.model_info = SUPPORTED_MODELS.get(
            self.model_name,
            {
                "dim": 768,
                "description": "Custom model",
                "prefix_query": "",
                "prefix_passage": "",
            },
        )
        self._model = None

    @property
    def dimension(self) -> int:
        return self.model_info["dim"]

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            cache_dir = os.getenv("HF_HOME", "/tmp/huggingface_cache")
            # Pin the device explicitly. Without it SentenceTransformer silently
            # took cuda:0. The accelerator module resolves cuda, mps or cpu, so
            # the same code works on every host.

            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self._model = SentenceTransformer(self.model_name, cache_folder=cache_dir, device=device)
            logger.info(f"Embedding model loaded: dim={self.dimension} device={device}")
        except ImportError:
            raise RuntimeError(
                "The sentence-transformers package is required. Run 'pip install sentence-transformers'."
            )

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed a list of texts."""
        self._load_model()
        prefix = self.model_info["prefix_query"] if is_query else self.model_info["prefix_passage"]
        if prefix:
            texts = [prefix + t for t in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embed_texts([query], is_query=True)[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embed_texts(documents, is_query=False)

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "description": self.model_info.get("description", ""),
            "loaded": self._model is not None,
        }

    @staticmethod
    def list_supported_models() -> list:
        return [{"name": name, **info} for name, info in SUPPORTED_MODELS.items()]
