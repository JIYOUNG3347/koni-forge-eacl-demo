"""The training target has to reach the tool arguments, or the run cannot start::

    {"error": "start_training_job is missing required parameters: base_model, dataset"}

The agent had written the exact values into its own final answer
(``Qwen--Qwen2.5-7B-Instruct`` / ``agent-gen-1788341983_twist``). It did not
fail for lack of knowledge, but because it never copied them into the tool
arguments — which were the only channel.

Three defects overlapped:
  (a) the fallback was dead — nothing ever assigned ``_last_dataset``
  (b) the working channel (``set_train_overrides``) only carried hyperparameters
  (c) the required-value check ran before the overrides were merged

What this file pins is mainly (a): a channel with a reader and no writer is
silent — no exception, no log, just an empty value every time.
"""

import re
from pathlib import Path

import pytest

from server.core.train_job_params import resolve_training_params

REPO = Path(__file__).resolve().parents[2]
DISPATCHER = REPO / "modules" / "agents" / "dispatcher.py"
TUNING = REPO / "modules" / "agents" / "specialists" / "tuning_specialist.py"


# ── Fallback semantics: fill in, never overwrite ─────────────────────────────


def test_fills_both_identity_values_when_missing():
    out = resolve_training_params({}, fallback_dataset="ds_twist", fallback_base_model="Qwen--Qwen2.5-7B-Instruct")
    assert out["ok"]
    assert out["params"]["dataset"] == "ds_twist"
    assert out["params"]["base_model"] == "Qwen--Qwen2.5-7B-Instruct"
    assert out["params"]["job_name"] == "ds_twist_training"


def test_does_not_overwrite_values_the_llm_supplied():
    """Different from apply_train_overrides, which does overwrite."""
    out = resolve_training_params(
        {"dataset": "chosen_ds", "base_model": "chosen_model"},
        fallback_dataset="pipeline_ds",
        fallback_base_model="pipeline_model",
    )
    assert out["params"]["dataset"] == "chosen_ds"
    assert out["params"]["base_model"] == "chosen_model"


def test_still_fails_honestly_when_nothing_is_known():
    """Never start training on a guess — this runs for hours."""
    out = resolve_training_params({})
    assert not out["ok"]
    assert sorted(out["missing"]) == ["base_model", "dataset"]
    for field in ("base_model", "dataset"):
        assert field in out["message"]


def test_blank_and_whitespace_fallbacks_are_not_accepted():
    out = resolve_training_params({}, fallback_dataset="   ", fallback_base_model="")
    assert not out["ok"]


# ── Wiring: a reader with no writer dies quietly ─────────────────────────────


def test_identity_channel_has_a_writer():
    """Preventing (a) from coming back — without this check the dead fallback lived on."""
    body = DISPATCHER.read_text(encoding="utf-8")
    assert "def set_train_identity" in body, "the dispatcher has no setter"
    assert "agent._pipeline_train_identity = " in body, (
        "the setter does not propagate to the agents — assigning the field after construction does not"
    )


def test_identity_channel_has_a_reader():
    body = TUNING.read_text(encoding="utf-8")
    assert "_pipeline_train_identity" in body, "the specialist never reads the channel"
    for field in ("dataset", "base_model"):
        assert f'_pipeline_identity_hint("{field}")' in body, f"the {field} fallback is not wired"


def test_setter_is_called_where_the_values_are_known():
    """The preprocessing handoff (the twist branch) is the one place that fixes both values."""
    body = DISPATCHER.read_text(encoding="utf-8")
    assert "self.set_train_identity(" in body, "nobody calls the setter"


def test_dead_attributes_are_gone():
    """Reading an attribute nothing assigns makes the fallback die quietly."""
    body = TUNING.read_text(encoding="utf-8")
    for dead in ("_last_dataset", "_pipeline_dataset"):
        assert not re.search(rf'getattr\(self, "{dead}"', body), (
            f"{dead} is read again — nothing anywhere assigns this attribute"
        )


@pytest.mark.parametrize("field", ["dataset", "base_model"])
def test_setter_does_not_clobber_with_blanks(field):
    """Each stage knows different things, so a partial update is normal."""
    body = DISPATCHER.read_text(encoding="utf-8")
    block = body[body.index("def set_train_identity") : body.index("@property\n    def agents")]
    assert f"if {field}:" in block, f"{field} is overwritten even when empty"
