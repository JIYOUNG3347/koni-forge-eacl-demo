"""The prompt and the parser are contracted on the same labels.

The dispatcher tells the LLM a markdown format and then reads that very output
back with a label regex. Nothing raises when the contract breaks: a failed
regex silently falls back to the ctx defaults, and the user is shown a run that
looks as if it followed the recommendation.

So this file weighs three things:
  1. values really are extracted from a recommendation card
  2. an unresolvable model is left empty instead of guessed
  3. the prompt and parser labels cannot drift apart
"""

from pathlib import Path

from server.core.rec_params import parse_rec_params

REPO = Path(__file__).resolve().parents[2]

_CTX = {
    "base_models": [{"name": "Qwen--Qwen3-4B"}, {"name": "google--gemma-3-1b"}],
    "rec_method": "lora",
    "rec_model": "",
    "rec_epochs": 3,
    "rec_batch": 4,
    "rec_lr": "2e-5",
}

_REC = """#### Recommended training setup
- **Base model**: Qwen3-4B (based on KBD coverage)
- **Method**: FFT (coverage is low at 20%)
- **Dataset**: dataset_x (500 samples)
- **Epochs**: 5
- **Batch size**: 8
- **Learning rate**: 2e-4 (small dataset)
- **BiLT**: enabled"""

_EXPECTED = {
    "rec_method": "sft",  # FFT is "sft" internally
    "rec_model": "Qwen--Qwen3-4B",
    "rec_epochs": 5,
    "rec_batch": 8,
    "rec_lr": "2e-4",
}


class TestParsing:
    def test_a_recommendation_card_parses(self):
        assert parse_rec_params(_REC, _CTX) == _EXPECTED

    def test_missing_fields_fall_back_to_ctx(self):
        out = parse_rec_params("#### Recommended training setup\n(nothing)", _CTX)
        assert out["rec_epochs"] == 3 and out["rec_lr"] == "2e-5"

    def test_case_insensitive(self):
        text = _REC.replace("Learning rate", "LEARNING RATE").replace("Epochs", "epochs")
        out = parse_rec_params(text, _CTX)
        assert out["rec_lr"] == "2e-4" and out["rec_epochs"] == 5

    def test_fullwidth_colon_accepted(self):
        """Local models sometimes mix full-width punctuation into their output."""
        out = parse_rec_params("- **Learning rate**： 3e-5", _CTX)
        assert out["rec_lr"] == "3e-5"

    def test_unresolvable_model_returns_empty_not_a_guess(self):
        out = parse_rec_params("- **Base model**: SomeUnknownModel", _CTX)
        assert out["rec_model"] == ""


class TestPromptParserContract:
    """If the prompt and the parser each spell the labels themselves, they drift."""

    def test_dispatcher_delegates_to_the_pure_module(self):
        src = (REPO / "modules/agents/dispatcher.py").read_text(encoding="utf-8")
        assert "from server.core.rec_params import parse_rec_params" in src
        assert "return parse_rec_params(rec_text, ctx, lang)" in src
        # A parsing body left in the dispatcher would let the two diverge.
        assert "Learning rate[^:：]" not in src, "the regex is still in the dispatcher"


# ── Nothing to train on ─────────────────────────────────────────────────────
# Without QA generation a dataset arrives only by upload, so an empty
# dataset_name must not become a card offering to start training.


def _ctx(**over):
    base = {
        "base_models": [{"name": "m", "size": " (1B)"}],
        "gpu_info": "gpu",
        "kbd_model": "m",
        "dataset_name": "",
        "dataset_count": 0,
        "rec_model": "m",
        "rec_method": "lora",
        "rec_epochs": 3,
        "rec_batch": 4,
        "rec_lr": "2e-4",
        "kbd_recommendation": "Fine-Tuning",
        "kbd_coverage_pct": 17.5,
    }
    base.update(over)
    return base


def test_the_prompt_says_there_is_no_dataset():
    from server.core.train_rec_prompt import build_prompt

    prompt = build_prompt(_ctx(), low_coverage_pct=30, default_max_seq_len=1024)
    assert "none available" in prompt
    assert "do not propose starting training" in prompt.lower()


def test_a_real_dataset_still_gets_its_row():
    from server.core.train_rec_prompt import build_prompt

    prompt = build_prompt(
        _ctx(dataset_name="ds1", dataset_count=120), low_coverage_pct=30, default_max_seq_len=1024
    )
    assert "ds1 (120 samples)" in prompt
    assert "none available" not in prompt


def test_the_fallback_card_does_not_invent_a_name():
    from server.core.train_rec_prompt import fallback_card

    assert "Dataset**: none" in fallback_card(_ctx())
