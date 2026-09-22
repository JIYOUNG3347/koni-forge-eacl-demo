"""The embedding model must be pinned to the detected device.

Left to itself SentenceTransformer picks cuda:0, so the device is passed
explicitly and the same code runs on CUDA, MPS and CPU. sentence_transformers
and torch are injected as fakes, so the test needs neither installed.
"""

import sys
import types

from modules.rag.embeddings import EmbeddingManager


def _install_fakes(monkeypatch, cuda_available: bool):
    captured = {}

    class FakeST:
        def __init__(self, model_name, cache_folder=None, device=None):
            captured["model_name"] = model_name
            captured["device"] = device

    fake_st_mod = types.ModuleType("sentence_transformers")
    setattr(fake_st_mod, "SentenceTransformer", FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "cuda", types.SimpleNamespace(is_available=lambda: cuda_available))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return captured


def test_device_cuda_when_available(monkeypatch):
    captured = _install_fakes(monkeypatch, cuda_available=True)
    em = EmbeddingManager(model_name="intfloat/multilingual-e5-base")
    em._load_model()
    assert captured["device"] == "cuda"


def test_device_cpu_when_no_cuda(monkeypatch):
    captured = _install_fakes(monkeypatch, cuda_available=False)
    em = EmbeddingManager(model_name="intfloat/multilingual-e5-base")
    em._load_model()
    assert captured["device"] == "cpu"


def test_device_cpu_when_torch_missing(monkeypatch):
    """A failed torch import still falls back to cpu without raising."""
    captured = {}

    class FakeST:
        def __init__(self, model_name, cache_folder=None, device=None):
            captured["device"] = device

    fake_st_mod = types.ModuleType("sentence_transformers")
    setattr(fake_st_mod, "SentenceTransformer", FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → raises

    em = EmbeddingManager(model_name="intfloat/multilingual-e5-base")
    em._load_model()
    assert captured["device"] == "cpu"
