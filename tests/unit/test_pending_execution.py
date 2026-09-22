"""A blocked execution has to surface as a modal, whichever way it was requested.

In guided mode an execution tool is blocked until the user confirms
(`_GUARDED_TOOLS`), but what happened after the block used to depend on the
entry path: asking in chat only repeated the question in text, while arriving
through a chain filled `pending_chain_params` and did open a parameter modal.
The same job looked different depending on how it was requested.

`selection_payload` reshapes the pending execution for the modal, and
`merge_overrides` writes the user's edits back through an allowlist.
"""

from pathlib import Path

from server.core.pending_execution import (
    EDITABLE_FIELDS,
    EDITABLE_TOOLS,
    is_editable,
    merge_overrides,
    selection_payload,
)

_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PENDING = {
    "tool_name": "start_training_job",
    "tool_input": {"base_model": "Qwen--Qwen3-4B", "epochs": 3},
}


def test_training_surfaces_a_modal_payload():
    payload = selection_payload("tuning", TRAIN_PENDING)
    assert payload == {
        "tool": "start_training_job",
        "agent": "tuning",
        "stage": "tuning",
        "params": {"base_model": "Qwen--Qwen3-4B", "epochs": 3},
    }


def test_the_editable_set_matches_the_agents():
    """An editable tool no agent exposes can never open its modal."""
    import re

    provided = set()
    for path in (_ROOT / "modules" / "agents" / "specialists").glob("*.py"):
        provided |= set(re.findall(r'"name":\s*"([a-z_]+)"', path.read_text(encoding="utf-8")))
    assert set(EDITABLE_TOOLS) <= provided, sorted(set(EDITABLE_TOOLS) - provided)


def test_read_only_tools_are_not_editable():
    """Only tools that start work get a parameter modal."""
    for tool in ("list_datasets", "get_gpu_info", "list_models"):
        assert is_editable(tool) is False
        assert selection_payload("tuning", {"tool_name": tool, "tool_input": {}}) is None


def test_no_pending_means_no_payload():
    for pend in (None, {}, "a string", {"tool_name": ""}, {"tool_name": "list_datasets"}):
        assert selection_payload("tuning", pend) is None  # type: ignore[arg-type]


def test_empty_params_still_opens_the_modal():
    """"nothing to edit" and "no modal" are different, and the latter was the bug."""
    payload = selection_payload("tuning", {"tool_name": "start_training_job"})
    assert payload is not None and payload["params"] == {}


def test_payload_does_not_alias_the_pending_input():
    """Editing the dict handed to the modal would silently change the pending state."""
    payload = selection_payload("tuning", TRAIN_PENDING)
    assert payload is not None
    params: dict = payload["params"]
    params["epochs"] = 99
    original: dict = TRAIN_PENDING["tool_input"]  # type: ignore[assignment]
    assert original["epochs"] == 3


# ── Merging ─────────────────────────────────────────────────────────────────


def test_only_whitelisted_keys_are_merged():
    """Merging arbitrary keys passes arguments outside the tool contract."""
    out = merge_overrides(
        "start_training_job",
        {"epochs": 3},
        {"epochs": 10, "shell": "rm -rf /", "__class__": "x"},
    )
    assert out == {"epochs": 10}


def test_none_means_unchanged():
    """The modal sends an emptied field as None; writing it would erase the value."""
    out = merge_overrides("start_training_job", {"method": "lora"}, {"method": None})
    assert out == {"method": "lora"}


def test_unknown_tool_merges_nothing():
    assert merge_overrides("mystery", {"a": 1}, {"a": 2}) == {"a": 1}


def test_merge_does_not_mutate_the_input():
    original = {"epochs": 3}
    merge_overrides("start_training_job", original, {"epochs": 10})
    assert original == {"epochs": 3}


def test_every_editable_tool_declares_its_fields():
    """Without a field list a tool can edit nothing, which makes the modal pointless."""
    assert set(EDITABLE_TOOLS) == set(EDITABLE_FIELDS)
    for tool, fields in EDITABLE_FIELDS.items():
        assert fields, tool


# ── Wiring guard ────────────────────────────────────────────────────────────


def test_status_exposes_the_payload():
    src = (_ROOT / "modules" / "agents" / "dispatcher.py").read_text(encoding="utf-8")
    assert '"pending_execution": self._pending_execution_payload()' in src
    assert "def _pending_execution_payload" in src


def test_confirm_endpoint_merges_through_the_allowlist():
    src = (_ROOT / "server" / "routers" / "agent.py").read_text(encoding="utf-8")
    assert '@router.post("/execution/confirm")' in src
    # Without the allowlist merge, arbitrary keys get through.
    assert "merge_overrides(" in src
    # Cancel does not execute.
    block = src[src.index('@router.post("/execution/confirm")') :][:2600]
    assert 'req.action == "cancel"' in block
    assert "_pending_execution = None" in block
