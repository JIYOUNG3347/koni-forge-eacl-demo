"""The confirm and skip buttons send their own label, and the matcher must accept it.

The buttons post the on-screen label verbatim — "Proceeding", "Skipping". The
vocabulary pack only held the stems (``proceed``, ``skip``), so the present
participle never matched and guided mode could not be advanced by its buttons.

Two contracts are pinned here:
  1. every value a button sends is read as the intent it belongs to, and
  2. no value is read as both, or there is no telling which way it goes.
"""

from __future__ import annotations

from pathlib import Path

from server.core.agent_intent_words import is_confirmation, is_skip

REPO = Path(__file__).resolve().parents[2]
AGENT_STORE = REPO / "UI" / "src" / "stores" / "agentStore.ts"

#: What the UI buttons actually send, and the intent each one must be read as.
BUTTON_VALUES = [
    ("Proceeding", "confirm"),
    ("Proceeding with the edited parameters", "confirm"),
    ("Skipping", "skip"),
]


def test_every_button_value_is_matched():
    """A value the button sends but the matcher rejects leaves guided mode stuck."""
    for value, intent in BUTTON_VALUES:
        matched = is_confirmation(value) if intent == "confirm" else is_skip(value)
        assert matched, f"the matcher does not read {value!r} as {intent}"


def test_confirmation_and_skip_stay_disjoint():
    """A value read as both leaves no way to know which path it takes."""
    for value, _ in BUTTON_VALUES:
        assert not (is_confirmation(value) and is_skip(value)), value


def test_the_button_values_still_match_the_ui():
    """If the pack and the UI drift apart, the buttons stop working again."""
    store = AGENT_STORE.read_text(encoding="utf-8")
    for value, _ in BUTTON_VALUES:
        assert f'"{value}"' in store, value


def test_dispatcher_no_longer_hardcodes_stage_labels():
    body = (REPO / "modules" / "agents" / "dispatcher.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    assert "_AGENT_LABELS" not in code
    assert "def stage_label(" in code
