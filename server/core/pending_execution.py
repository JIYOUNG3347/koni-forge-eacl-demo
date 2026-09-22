"""Turn a blocked execution into something the confirmation modal can show.

In guided mode an execution tool is blocked until the user confirms
(``foundation._GUARDED_TOOLS``). This reshapes the pending execution for the
modal and merges the user's edits back, **allowlisted keys only** — merging
arbitrary keys would pass arguments outside the tool contract.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

#: Execution tools that open a confirmation modal, mapped to the modal's stage name.
EDITABLE_TOOLS: Dict[str, str] = {
    "start_training_job": "tuning",
}

#: Per-tool arguments the modal may edit. This is an allowlist; other keys are ignored,
#: because merging arbitrary keys passes arguments outside the tool contract.
EDITABLE_FIELDS: Dict[str, frozenset] = {
    "start_training_job": frozenset({"job_name", "base_model", "dataset", "epochs", "learning_rate", "method"}),
}


def is_editable(tool_name: Optional[str]) -> bool:
    """Whether this tool opens a confirmation modal."""
    return str(tool_name or "") in EDITABLE_TOOLS


#: Every tool that actually shows a modal when the guard blocks it.
#: `download_model` is absent on purpose: the guard catches it but there is no
#: modal, so blocking only repeats the question in text and the user can never
#: get past "yes -> asked again".
MODAL_TOOLS: frozenset = frozenset(EDITABLE_TOOLS)


def has_modal(tool_name: Optional[str]) -> bool:
    """Whether blocking this tool opens a modal."""
    return str(tool_name or "") in MODAL_TOOLS


def should_block(tool_name: Optional[str], *, confirmed: bool, params_seen: bool) -> bool:
    """Two inputs decide this:

    * ``confirmed`` — the user expressed approval
    * ``params_seen`` — that approval came *after* the modal (a resume)

    A chat "yes" sets only the first, so a tool with a modal is blocked once to
    show its parameters. Approving in the modal enters through resume and runs.
    """
    if not confirmed:
        return True
    if params_seen:
        return False
    return has_modal(tool_name)


def selection_payload(agent_name: str, pending: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pending execution to a modal payload. None when the tool has no modal.

    A payload is built even with empty ``params``: showing "nothing to edit" is
    different from showing no modal at all, and the latter was the bug.
    """
    if not isinstance(pending, Mapping):
        return None
    tool = str(pending.get("tool_name") or "")
    if not is_editable(tool):
        return None
    raw = pending.get("tool_input")
    params = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "tool": tool,
        "agent": agent_name,
        "stage": EDITABLE_TOOLS[tool],
        "params": params,
    }


def merge_overrides(
    tool_name: str, tool_input: Optional[Mapping[str, Any]], overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Merge the user's edits into the pending arguments.

    Allowlisted keys only. ``None`` means "unchanged" and is skipped — the modal
    sends an emptied field as ``None``, and writing it would erase the value.
    """
    merged: Dict[str, Any] = dict(tool_input or {})
    allowed = EDITABLE_FIELDS.get(str(tool_name or ""), frozenset())
    for key, value in (overrides or {}).items():
        if key in allowed and value is not None:
            merged[key] = value
    return merged
