"""A chat "yes" must not skip the parameter modal.

The failure this guards against::

    [user]      Train on dataset_20260908_102036.
    [assistant] (only read-only tools called) ... Shall I proceed with these
                settings?                          <- never tried the real tool
    [user]      yes
    -> start_training_job ran with parameters the user never saw

Nothing was pending after the first turn, so no modal could open — correct. On
the second turn ``_execution_confirmed`` was set and skipped the guard outright,
and the job started without the user ever seeing the parameters.

A tool with no modal is deliberately *not* blocked: ``download_model`` hits the
guard but has nothing to show, so blocking it just repeats the same question
forever. That distinction is what this pins.
"""

import ast
import re
from pathlib import Path

import pytest

from server.core.confirm_signal import is_confirmation_question
from server.core.pending_execution import (
    EDITABLE_TOOLS,
    MODAL_TOOLS,
    has_modal,
    should_block,
)

REPO = Path(__file__).resolve().parents[2]

#: Every tool the guard blocks in guided mode.
_GUARDED = ("start_training_job", "download_model")
#: Blocked, but with no modal to show.
_NO_MODAL = ("download_model",)


# ── the decision table ───────────────────────────────────────────────
class TestShouldBlock:
    @pytest.mark.parametrize("tool", _GUARDED)
    @pytest.mark.parametrize("seen", [False, True])
    def test_unconfirmed_always_blocks(self, tool, seen):
        assert should_block(tool, confirmed=False, params_seen=seen) is True

    @pytest.mark.parametrize("tool", _GUARDED)
    def test_resume_never_blocks(self, tool):
        """Approval after seeing the modal passes, or the confirm button is dead."""
        assert should_block(tool, confirmed=True, params_seen=True) is False

    @pytest.mark.parametrize("tool", sorted(MODAL_TOOLS))
    def test_chat_yes_blocks_modal_tools(self, tool):
        """A chat "yes" is not the same as having seen the parameters."""
        assert should_block(tool, confirmed=True, params_seen=False) is True

    @pytest.mark.parametrize("tool", _NO_MODAL)
    def test_chat_yes_passes_tools_without_a_modal(self, tool):
        """With nothing to show, blocking would repeat the same question forever."""
        assert should_block(tool, confirmed=True, params_seen=False) is False

    def test_unknown_tool_is_not_blocked_after_confirmation(self):
        """Blocking an unknown tool would create a loop — when unsure, allow."""
        assert should_block("some_new_tool", confirmed=True, params_seen=False) is False
        assert should_block(None, confirmed=True, params_seen=False) is False


# ── which tools have a modal ─────────────────────────────────────────
class TestModalTools:
    def test_every_editable_tool_has_a_modal(self):
        assert MODAL_TOOLS == frozenset(EDITABLE_TOOLS)

    @pytest.mark.parametrize("tool", _NO_MODAL)
    def test_excludes_tools_without_a_modal(self, tool):
        assert not has_modal(tool)

    def test_the_guarded_set_matches_the_agents(self):
        """A guarded tool no agent exposes can never be reached — and blocking a
        tool the agents do expose would stall guided mode."""
        import re

        provided = set()
        for path in (REPO / "modules" / "agents" / "specialists").glob("*.py"):
            provided |= set(re.findall(r'"name":\s*"([a-z_]+)"', path.read_text(encoding="utf-8")))
        assert set(_GUARDED) <= provided, sorted(set(_GUARDED) - provided)
        assert set(EDITABLE_TOOLS) <= provided, sorted(set(EDITABLE_TOOLS) - provided)


class TestIncident:
    _ASSISTANT_ASK = "Shall I proceed with these settings?"

    def test_the_live_sequence_now_opens_the_modal(self):
        """The agent asks in text, the user says "yes" — now it blocks once."""
        assert is_confirmation_question(self._ASSISTANT_ASK) is True
        assert should_block("start_training_job", confirmed=True, params_seen=False) is True

    def test_the_flow_terminates(self):
        """Approving in the modal resumes and runs — this is not a loop."""
        tool = "start_training_job"
        # 1) chat "yes" is blocked and the modal opens
        assert should_block(tool, confirmed=True, params_seen=False) is True
        # 2) modal approval resumes the pending execution and passes
        assert should_block(tool, confirmed=True, params_seen=True) is False

    @pytest.mark.parametrize("tool", _NO_MODAL)
    def test_no_modal_tool_never_loops(self, tool):
        """A tool without a modal passes on the first "yes" — never asked twice."""
        assert should_block(tool, confirmed=True, params_seen=False) is False


# ── wiring: both paths use the same decision ─────────────────────────
class TestWiring:
    _SRC = (REPO / "modules/agents/foundation.py").read_text(encoding="utf-8")

    def test_resume_marks_params_as_seen(self):
        assert "self._execution_params_seen = True" in self._SRC

    def test_chat_path_marks_params_as_unseen(self):
        assert "self._execution_params_seen = False" in self._SRC

    def test_every_confirmed_assignment_sets_the_companion_flag(self):
        """Wherever ``_execution_confirmed`` moves, its companion flag must too.

        Leaving one behind lets a previous run's value leak into the next, which
        is exactly the old behaviour where a chat "yes" silently skipped the modal.
        """
        tree = ast.parse(self._SRC)
        confirmed_lines, seen_lines = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == "self" and target.attr == "_execution_confirmed":
                        confirmed_lines.append(node.lineno)
                    if target.value.id == "self" and target.attr == "_execution_params_seen":
                        seen_lines.append(node.lineno)
        assert confirmed_lines, "no decision flag found — the guard reads nothing"
        for line in confirmed_lines:
            assert any(abs(line - s) <= 4 for s in seen_lines), (
                f"foundation.py:{line} changes `_execution_confirmed` without "
                f"`_execution_params_seen` — the value leaks into the next run"
            )


# ── the modal's approve button must wake the resume path ─────────────
#: The inline confirmation words in ``foundation.run``, used by the resume path.
_FOUNDATION_CONFIRM_WORDS = {"yes", "y", "ok", "okay", "go", "proceed", "start", "sure"}


def test_modal_approve_wakes_the_resume_path():
    """If this breaks, the result is an infinite loop.

    Every chat-requested execution now passes through the modal once. If the
    modal's approve text does not match ``foundation``'s confirmation words, the
    pending execution never wakes and the same question repeats forever.

    ``test_chain_advance_action`` only checks ``agent_intent_words.is_confirmation``,
    which is a different function from the set below, so this is pinned separately.
    """
    from server.core.agent_runtime_texts import text

    approve = text("chain_advance_approve", "en")
    lowered = approve.strip().lower().rstrip(".!?")
    hits = sorted(w for w in _FOUNDATION_CONFIRM_WORDS if w in lowered)
    assert hits, f"approve text {approve!r} does not wake the resume path — this loops"


def test_foundation_confirm_words_match_the_pack():
    """The same vocabulary exists twice; if they diverge the invariant above breaks."""
    from server.core.confirm_signal import CONFIRM_WORDS

    src = (REPO / "modules/agents/foundation.py").read_text(encoding="utf-8")
    assert src.count("_confirm_words = {") == 1, "a second inline set would make three copies"
    for word in CONFIRM_WORDS:
        assert f'"{word}"' in src, f"confirmation word {word!r} is missing from the foundation set"


# ── purity: the decision must run in a plain CI venv ─────────────────
def test_decision_module_stays_pure():
    """If the decision leaks into routers or celery, unit tests cannot reach it."""
    src = (REPO / "server/core/pending_execution.py").read_text(encoding="utf-8")
    for forbidden in ("import fastapi", "import celery", "import torch", "from fastapi", "from celery"):
        assert forbidden not in src, f"heavy dependency in a pure module: {forbidden}"
    assert not re.search(r"^\s*import\s+(redis|chromadb)", src, re.M)
