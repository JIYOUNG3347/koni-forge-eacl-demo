from pathlib import Path

from server.core.llm_tokens import (
    DECISION_TOKENS,
    JSON_TOKENS,
    NARRATIVE_TOKENS,
    REASONING_HEADROOM,
    SCORE_TOKENS,
    with_headroom,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_headroom_covers_measured_failure_threshold():
    assert REASONING_HEADROOM >= 800


def test_with_headroom_adds_thinking_room():
    assert with_headroom(16) == 16 + REASONING_HEADROOM
    assert with_headroom(512) == 512 + REASONING_HEADROOM


def test_with_headroom_handles_bad_input():
    assert with_headroom(0) >= REASONING_HEADROOM
    assert with_headroom(-5) >= REASONING_HEADROOM
    assert with_headroom("abc") >= REASONING_HEADROOM  # type: ignore[arg-type]


def test_all_presets_exceed_failure_threshold():
    for budget in (SCORE_TOKENS, DECISION_TOKENS, JSON_TOKENS, NARRATIVE_TOKENS):
        assert budget >= 800


def test_narrative_is_largest():
    # A narrative (three or four sentences) gets a bigger budget than one score.
    assert NARRATIVE_TOKENS > JSON_TOKENS > DECISION_TOKENS > SCORE_TOKENS




def test_dispatcher_no_longer_uses_512():
    src = (PROJECT_ROOT / "modules/agents/dispatcher.py").read_text(encoding="utf-8")
    assert "complete_text(prompt, max_tokens=512)" not in src
