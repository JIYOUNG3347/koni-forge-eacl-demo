"""Only the pure `decide_skip` is tested, so AgentDispatcher is never
instantiated and this passes without openai, torch and the rest.
"""

from modules.agents.dispatcher import (
    PIPELINE_STAGES,
    decide_skip,
)

# ── A. decide_skip per proposed agent ─────────────────────────────────


def test_skip_assessment_ends():
    """assessment is terminal — there is nothing left to skip to."""
    d = decide_skip("assessment")
    assert d["action"] == "end"
    assert d["land"] is None


def test_skip_none_or_unknown_ends_gracefully():
    """None (an RAG terminal) or an unknown agent ends gracefully."""
    assert decide_skip(None)["action"] == "end"
    assert decide_skip("nonexistent")["action"] == "end"


# ── C. The legacy skip_to is entirely gone ────────────────────────────
def test_no_skip_to_keys_remain():
    """Static skip_to jumps are retired and must not remain on any stage."""
    for stage, rule in PIPELINE_STAGES.items():
        assert "skip_to" not in rule, f"skip_to left on {stage}"


# ── D. The block message ─────────────────────────────────────────────
