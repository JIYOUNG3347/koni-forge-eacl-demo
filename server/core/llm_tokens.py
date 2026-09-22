"""Token budget: expected output size plus reasoning headroom.

External models without a thinking phase never use the headroom, which is
harmless — ``max_tokens`` is a ceiling, and billing follows what is generated.
"""

from __future__ import annotations

REASONING_HEADROOM = 1024

# Expected output size per use, before headroom.
EXPECTED_SCORE = 16  # one number (e.g. faithfulness 0.87)
EXPECTED_DECISION = 64  # a short structured verdict (e.g. {"action": "chat"})
EXPECTED_JSON = 256  # medium JSON, such as hyperparameters
EXPECTED_NARRATIVE = 512  # three or four sentences of prose


def with_headroom(expected_output_tokens: int, headroom: int = REASONING_HEADROOM) -> int:
    """Expected output plus reasoning headroom equals the real ``max_tokens``.

    Args:
        expected_output_tokens: roughly how many tokens the answer body needs.
        headroom: thinking allowance (default :data:`REASONING_HEADROOM`).
    """
    try:
        expected = int(expected_output_tokens)
    except (TypeError, ValueError):
        expected = 0
    return max(1, expected) + max(0, headroom)


# Common combinations, so callers read as intent.
SCORE_TOKENS = with_headroom(EXPECTED_SCORE)
DECISION_TOKENS = with_headroom(EXPECTED_DECISION)
JSON_TOKENS = with_headroom(EXPECTED_JSON)
NARRATIVE_TOKENS = with_headroom(EXPECTED_NARRATIVE)


