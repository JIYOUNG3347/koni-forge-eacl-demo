"""Order matters when releasing device memory after training.

Measured after a run:
    right after training        6175MB
    empty_cache() then gc()     2489MB
    gc() then empty_cache()     2447MB   <- the order fix alone: 2%
    plus expandable_segments     567MB   <- together: 77%

It is not a reference leak: no live tensors remain. The two changes are a pair,
so both are pinned here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cleanup_body() -> str:
    """Body of ``_cleanup_gpu_memory`` without its docstring.

    The docstring names ``empty_cache()`` as part of the rationale, so reading
    the whole function would make the order check match the prose instead.
    """
    src = (ROOT / "celery_app" / "tasks" / "train_tasks.py").read_text(encoding="utf-8")
    start = src.index("def _cleanup_gpu_memory(")
    end = src.index("\ndef ", start + 1)
    func = src[start:end]
    doc_open = func.index('"""')
    doc_close = func.index('"""', doc_open + 3) + 3
    return func[doc_close:]


def test_gc_runs_before_empty_cache():
    """empty_cache() only returns blocks that are already fully free."""
    body = _cleanup_body()
    assert "gc.collect()" in body
    assert "empty_cache()" in body
    assert body.index("gc.collect()") < body.index("empty_cache()"), (
        "gc.collect() must run before empty_cache(), or one call in finally frees nothing"
    )


def test_cuda_path_synchronizes_after_empty_cache():
    """The CUDA branch of the shared helper syncs after emptying the cache."""
    src = (ROOT / "server" / "core" / "accelerator.py").read_text(encoding="utf-8")
    body = src[src.index("def empty_cache("):]
    assert body.index("torch.cuda.empty_cache()") < body.index("torch.cuda.synchronize()")


def test_expandable_segments_is_declared_in_env_example():
    """The allocator setting carries most of the win; the order fix alone was 2%."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_ALLOC_CONF" in example
    assert "expandable_segments:True" in example
