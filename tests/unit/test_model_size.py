"""What this covers:
- weight file identification: safetensors and bin only, optimizer and scheduler state excluded
- dtype: torch_dtype/dtype in config.json to bytes per parameter, bf16 (2.0) when unknown
- bytes to billions, and when the answer is "unknown" (None)

torch-free pure module, so it runs in a plain CI venv.
"""

import server.core.model_size as ms


# ── Weight file identification ──────────────────────────────
def test_weight_files_recognised():
    assert ms.is_weight_file("model-00001-of-00004.safetensors") is True
    assert ms.is_weight_file("pytorch_model.bin") is True


def test_optimizer_state_is_not_weight():
    # Resume state, not model size. Including it inflates a checkpoint two or three times.
    for name in ("optimizer.pt", "optimizer.bin", "scheduler.pt", "training_args.bin"):
        assert ms.is_weight_file(name) is False


def test_non_weight_extensions_excluded():
    for name in ("config.json", "tokenizer.model", "README.md", "trainer_state.json"):
        assert ms.is_weight_file(name) is False


def test_sum_ignores_non_weights():
    entries = [
        ("model.safetensors", 1_000),
        ("optimizer.pt", 9_999_999),  # must be excluded
        ("config.json", 500),
    ]
    assert ms.sum_weight_bytes(entries) == 1_000


def test_sum_handles_bad_sizes():
    assert ms.sum_weight_bytes([("model.safetensors", None), ("a.bin", "x")]) == 0  # type: ignore[list-item]
    assert ms.sum_weight_bytes([("model.safetensors", -5)]) == 0


# ── dtype ───────────────────────────────────────────────────
def test_dtype_from_config():
    assert ms.bytes_per_param({"torch_dtype": "bfloat16"}) == 2.0
    assert ms.bytes_per_param({"torch_dtype": "float32"}) == 4.0
    assert ms.bytes_per_param({"dtype": "float16"}) == 2.0


def test_dtype_defaults_to_bf16():
    for cfg in (None, {}, {"torch_dtype": None}, {"torch_dtype": "weird-dtype"}, {"torch_dtype": 16}):
        assert ms.bytes_per_param(cfg) == ms.DEFAULT_BYTES_PER_PARAM == 2.0


# ── Conversion ──────────────────────────────────────────────
def test_params_b_bf16():
    # An 8B model in bf16 is about 16GB.
    got = ms.params_b_from_bytes(16_000_000_000, 2.0)
    assert got is not None and round(got, 1) == 8.0


def test_params_b_fp32_halves_estimate():
    # The same bytes in fp32 mean half the parameters.
    got = ms.params_b_from_bytes(16_000_000_000, 4.0)
    assert got is not None and round(got, 1) == 4.0


def test_params_b_unknown_returns_none():
    # None means "unknown", and the caller then takes the safe path.
    assert ms.params_b_from_bytes(0) is None
    assert ms.params_b_from_bytes(-1) is None
    assert ms.params_b_from_bytes(1000, 0) is None
    assert ms.params_b_from_bytes("x") is None  # type: ignore[arg-type]


# ── End-to-end estimate ─────────────────────────────────────
def test_estimate_realistic_8b_checkpoint():
    entries = [
        ("model-00001-of-00002.safetensors", 8_000_000_000),
        ("model-00002-of-00002.safetensors", 8_000_000_000),
        ("optimizer.pt", 32_000_000_000),  # excluded
        ("config.json", 1_200),
    ]
    got = ms.estimate_params_b(entries, {"torch_dtype": "bfloat16"})
    assert got is not None and round(got, 1) == 8.0


def test_estimate_empty_dir_is_none():
    assert ms.estimate_params_b([]) is None
    assert ms.estimate_params_b([("config.json", 100)]) is None
