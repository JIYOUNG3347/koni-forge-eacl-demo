"""Two things in the confirmation pack are machine contracts.

1. **The parameter list** — the LLM copies it straight into a tool call.
2. **The confirmation words** — the hint tells the user what to type, and the
   matcher has to accept exactly that. If the two drift apart, answering as
   instructed does not proceed.
"""

import re
from pathlib import Path

from server.core.agent_confirm import (
    SUGGESTED_CONFIRM_WORDS,
    TEXTS,
    TOOL_TOKENS,
    texts,
)
from server.core.agent_intent_words import is_confirmation

REPO = Path(__file__).resolve().parents[2]
DISPATCHER = REPO / "modules/agents/dispatcher.py"


class TestConfirmWordContract:
    """A word the hint suggests must be one the matcher accepts."""

    def test_suggested_words_are_accepted_by_the_matcher(self):
        """The contract is not that the word appears in a list, but that the
        matcher actually reads it as a confirmation."""
        for lang, words in SUGGESTED_CONFIRM_WORDS.items():
            for word in words:
                assert is_confirmation(word, lang), (
                    f"the {lang} hint suggests '{word}' but the matcher rejects it — "
                    f"answering as instructed would not proceed"
                )


class TestEnglish:
    def test_no_korean_remains(self):
        for key, value in TEXTS["en"].items():
            assert not re.search(r"[\uac00-\ud7a3]", value), f"Korean left in {key}"

    def test_tool_names_survive(self):
        """A translated tool name makes the LLM call something that does not exist."""
        joined = " ".join(TEXTS["en"].values())
        for token in TOOL_TOKENS:
            assert token in joined, f"{token} is missing from the English pack"

    def test_parameter_label_is_a_label_only(self):
        """Only the label is translated; the caller appends the parameters verbatim."""
        assert texts("en")["exact_params_label"].endswith(" ")
        assert "=" not in texts("en")["exact_params_label"]


class TestWiring:
    def test_no_confirmation_wording_left_in_dispatcher(self):
        """Wording in two places drifts — the dispatcher only consumes the pack."""
        src = DISPATCHER.read_text(encoding="utf-8")
        for phrase in (
            "The user confirmed the parameters below",
            "The user confirmed the recommended parameters",
            "Exact parameters:",
            "[Pipeline goal]",
        ):
            assert phrase not in src, f"'{phrase}' is hardcoded in the dispatcher"

    def test_dispatcher_uses_the_pack(self):
        src = DISPATCHER.read_text(encoding="utf-8")
        assert "_ct(self._exec_lang())" in src
        assert "_confirm_hint(self._exec_lang())" in src
