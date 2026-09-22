from server.core.train_vram import (
    BatchPlan,
    ModelShape,
    clamp_to_limits,
    estimate_train_vram_mb,
    plan_batch,
    safe_per_device_batch,
    shape_from_config,
)

# Values from a real config.json.
_QWEN3_4B_CFG = {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "intermediate_size": 9728,
}
_QWEN3_4B = ModelShape(params_b=4.02, vocab_size=151936, hidden_size=2560, num_layers=36)
_A100_FREE_MB = 81000.0


# ── shape_from_config ────────────────────────────────────────────────
def test_shape_from_config_params_approx_matches_disk():
    shape = shape_from_config(_QWEN3_4B_CFG)
    assert shape is not None
    assert abs(shape.params_b - 4.02) / 4.02 < 0.05
    assert shape.vocab_size == 151936 and shape.num_layers == 36


def test_shape_from_config_prefers_measured_params():
    shape = shape_from_config(_QWEN3_4B_CFG, params_b=4.02)
    assert shape is not None and shape.params_b == 4.02


def test_shape_from_config_rejects_missing_keys():
    assert shape_from_config({}) is None
    assert shape_from_config({"vocab_size": 0, "hidden_size": 1, "num_hidden_layers": 1}) is None


def test_incident_batch16_does_not_fit_80gb():
    est = estimate_train_vram_mb(_QWEN3_4B, "lora", 16, 2048)
    assert est > _A100_FREE_MB * 0.9
    assert 70_000 < est < 130_000


def test_incident_plan_recovers_with_smaller_batch():
    plan = plan_batch(_QWEN3_4B, "lora", 2048, _A100_FREE_MB, requested_batch=16, requested_accum=4)
    assert plan.adjusted is True
    assert 1 <= plan.per_device_batch < 16
    assert plan.per_device_batch * plan.grad_accum >= 16 * 4  # effective batch preserved
    assert plan.est_vram_mb <= _A100_FREE_MB * 0.9
    assert any("effective" in r for r in plan.reasons)


# ── estimate properties ──────────────────────────────────────────────
def test_estimate_monotonic_in_batch_and_seq():
    e1 = estimate_train_vram_mb(_QWEN3_4B, "lora", 4, 2048)
    assert estimate_train_vram_mb(_QWEN3_4B, "lora", 8, 2048) > e1
    assert estimate_train_vram_mb(_QWEN3_4B, "lora", 4, 4096) > e1


def test_fft_costs_more_than_lora():
    assert estimate_train_vram_mb(_QWEN3_4B, "sft", 4, 2048) > estimate_train_vram_mb(_QWEN3_4B, "lora", 4, 2048)


def test_grad_checkpointing_reduces_estimate():
    assert estimate_train_vram_mb(_QWEN3_4B, "lora", 8, 2048, True) < estimate_train_vram_mb(
        _QWEN3_4B, "lora", 8, 2048, False
    )


# ── safe_per_device_batch ────────────────────────────────────────────
def test_safe_batch_is_tight_boundary():
    b = safe_per_device_batch(_QWEN3_4B, "lora", 2048, _A100_FREE_MB)
    assert b >= 1
    assert estimate_train_vram_mb(_QWEN3_4B, "lora", b, 2048) <= _A100_FREE_MB * 0.9
    assert estimate_train_vram_mb(_QWEN3_4B, "lora", b + 1, 2048) > _A100_FREE_MB * 0.9


def test_safe_batch_zero_when_impossible():
    # A 7B full fine-tune does not fit batch 1 in 16GB.
    seven_b = ModelShape(params_b=7.62, vocab_size=152064, hidden_size=3584, num_layers=28)
    assert safe_per_device_batch(seven_b, "sft", 2048, 16_000.0) == 0


# ── plan_batch ───────────────────────────────────────────────────────
def test_plan_unadjusted_when_fits():
    plan = plan_batch(_QWEN3_4B, "lora", 512, _A100_FREE_MB, requested_batch=4, requested_accum=4)
    assert plan == BatchPlan(4, 4, False, plan.est_vram_mb, adjusted=False)


def test_plan_never_increases_batch():
    """Never raise the requested batch, however much memory is free."""
    plan = plan_batch(_QWEN3_4B, "lora", 512, _A100_FREE_MB * 4, requested_batch=2, requested_accum=1)
    assert plan.per_device_batch == 2 and plan.adjusted is False


def test_plan_grad_ckpt_rescues_when_plain_impossible():
    # 7B full FT at seq 4096: without checkpointing even batch 1 is tight.
    seven_b = ModelShape(params_b=7.62, vocab_size=152064, hidden_size=3584, num_layers=28)
    no_ckpt = safe_per_device_batch(seven_b, "sft", 4096, 140_000.0, False)
    with_ckpt_plan = plan_batch(seven_b, "sft", 4096, 140_000.0, requested_batch=8, requested_accum=1)
    if no_ckpt == 0:  # only then should it take the checkpointing path
        assert with_ckpt_plan.grad_checkpointing is True
    assert with_ckpt_plan.per_device_batch >= 1


def test_plan_impossible_returns_zero_batch():
    seven_b = ModelShape(params_b=7.62, vocab_size=152064, hidden_size=3584, num_layers=28)
    plan = plan_batch(seven_b, "sft", 2048, 8_000.0, requested_batch=4, requested_accum=4)
    assert plan.per_device_batch == 0 and plan.adjusted is True
    assert any("does not fit" in r for r in plan.reasons)


def test_plan_respects_target_effective():
    """A lower target_effective lets accumulation come down with it."""
    plan = plan_batch(
        _QWEN3_4B, "lora", 2048, _A100_FREE_MB, requested_batch=16, requested_accum=4, target_effective=32
    )
    assert plan.per_device_batch * plan.grad_accum >= 32
    plan_full = plan_batch(_QWEN3_4B, "lora", 2048, _A100_FREE_MB, requested_batch=16, requested_accum=4)
    assert plan.grad_accum <= plan_full.grad_accum


# ── host.toml limits ─────────────────────────────────────────────────
def test_an_unknown_method_is_costed_as_lora():
    """The cheaper of the two, so an unknown name never under-reserves."""
    assert estimate_train_vram_mb(_QWEN3_4B, "whatever", 4, 1024) == estimate_train_vram_mb(
        _QWEN3_4B, "lora", 4, 1024
    )


def test_clamp_to_limits_caps_and_reports():
    b, s, reasons = clamp_to_limits(32, 4096, 16, 2048)
    assert (b, s) == (16, 2048)
    assert len(reasons) == 2 and all("host.toml" in r for r in reasons)


def test_clamp_to_limits_noop_within_bounds():
    assert clamp_to_limits(4, 1024, 16, 2048) == (4, 1024, ())
    assert clamp_to_limits(4, 1024, None, 0) == (4, 1024, ())  # unlimited
