"""These strings are a contract, not display text.

The dispatcher embeds one in the chain message (`next_message`), and the
specialists' system prompts read it to decide how to behave::

    If the message contains "[pipeline auto run]":
    - The user has already confirmed
    - Do not ask again ("Shall I proceed?")
    - Immediately run in this order: list_models -> get_gpu_info -> ...

Every prompt pack references the same token, so the repository already treats
it as a machine token that is never translated — the same class as
`agent_handoff.TOOL_TOKENS` (`[navigate:...]`, `[chain_target:...]`).
That assumption was never enforced anywhere, which is what this module fixes.

**The test is an exact token, not a prefix.** The old check looked only at the
prefix, so the guided marker also matched — a test that cannot tell the two
modes apart was deciding "is this auto".
"""

from __future__ import annotations

from typing import Any

#: Auto: the user has already confirmed, so call the tool without asking.
AUTO_MARKER = "[pipeline auto run]"

#: Guided: gather information, recommend, and wait for the user.
GUIDED_MARKER = "[pipeline guided run]"

#: Never translate these. Same rule as `agent_handoff.TOOL_TOKENS`: the
#: strings must survive verbatim for all their consumers to keep matching.
MARKERS = (AUTO_MARKER, GUIDED_MARKER)


def _text(message: Any) -> str:
    return message if isinstance(message, str) else ""


def has_pipeline_marker(message: Any) -> bool:
    """Whether this is a pipeline chaining message, in either mode.

    Replaces the old prefix test. It matches the complete token, so a future
    marker with the same prefix cannot drag this decision along with it.

    """
    text = _text(message)
    return any(marker in text for marker in MARKERS)
