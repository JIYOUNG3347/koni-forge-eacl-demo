"""Alias to HuggingFace id normalisation.

EmbeddingManager.__init__ does not load the model (_load_model is lazy), so the
alias resolution can be checked without torch. Without this mapping the
indexer hands SentenceTransformer a repo that does not exist ("multilingual-e5-base")
"""

from modules.rag.embeddings import ALIAS_MAP, DEFAULT_MODEL, EmbeddingManager


def test_alias_resolves_to_full_hf_id():
    assert EmbeddingManager(model_name="multilingual-e5-base").model_name == "intfloat/multilingual-e5-base"
    assert EmbeddingManager(model_name="multilingual-e5-large").model_name == "intfloat/multilingual-e5-large"
    assert EmbeddingManager(model_name="ko-sroberta").model_name == "jhgan/ko-sroberta-multitask"
    assert EmbeddingManager(model_name="bge-m3").model_name == "BAAI/bge-m3"


def test_full_id_passes_through_unchanged():
    full = "intfloat/multilingual-e5-base"
    assert EmbeddingManager(model_name=full).model_name == full


def test_none_falls_back_to_default():
    # With RAG_EMBEDDING_MODEL unset, DEFAULT_MODEL (the full id) applies.
    assert EmbeddingManager(model_name=None).model_name in (
        DEFAULT_MODEL,
        ALIAS_MAP.get(DEFAULT_MODEL, DEFAULT_MODEL),
    )


def test_resolved_alias_has_correct_e5_prefix():
    """After alias resolution it must still match SUPPORTED_MODELS, e5 prefix and all."""
    em = EmbeddingManager(model_name="multilingual-e5-base")
    assert em.model_info["prefix_query"] == "query: "
    assert em.model_info["prefix_passage"] == "passage: "
