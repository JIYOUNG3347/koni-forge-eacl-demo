"""Pure functions only (route_match, is_question, classify_resume), so
AgentDispatcher is never instantiated and this passes in a plain CI venv.

Covered:
  A. route_match — stage keywords to an agent, None on a miss (distinct from
  B. is_question — the question heuristic, so a clarification does not run a specialist
  C. classify_resume — matched / override / mismatched / unmatched / no_pending
"""

from modules.agents.dispatcher import (
    classify_resume,
    is_question,
    route_match,
)

# ── A. route_match ───────────────────────────────────────────────────


def test_route_match_unmatched_returns_none():
    """A miss is None, distinct from the 'corpus' fallthrough in _route()."""
    assert route_match("hello there") is None
    assert route_match("xyzzy") is None


# ── B. is_question ───────────────────────────────────────────────────


def test_is_question_false_cases():
    assert is_question("run the KBD analysis") is False
    assert is_question("start training") is False
    assert is_question("generate the data") is False


# ── C. classify_resume ───────────────────────────────────────────────
def test_classify_no_pending():
    assert classify_resume("boundary", None, None) == "no_pending"


def test_classify_unmatched():
    assert classify_resume(None, "boundary", None) == "unmatched"


def test_classify_matched():
    assert classify_resume("boundary", "boundary", None) == "matched"


def test_classify_mismatched():
    assert classify_resume("tuning", "boundary", None) == "mismatched"


def test_classify_override_on_repeat():
    """Repeating the stage we just asked about forces it through."""
    assert classify_resume("tuning", "boundary", "tuning") == "override"


def test_classify_mismatched_when_override_differs():
    """A different mismatched stage is not an override — ask again."""
    assert classify_resume("tuning", "boundary", "preprocessing") == "mismatched"
