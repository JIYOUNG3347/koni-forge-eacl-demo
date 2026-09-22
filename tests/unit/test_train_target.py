"""Resolving which dataset a training request refers to.

The failure this guards against::

    dataset: Go (0 samples)
    epochs: 3 (To allow learning from scratch given zero samples)

``Go`` is not a dataset, and the LLM then reasoned from that zero to justify
its recommendation. A wrong target means the whole run trains on wrong data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from server.core.train_target import (
    dataset_names,
    model_handles,
    resolve_requested_model,
    select_training_dataset,
)

REPO = Path(__file__).resolve().parents[2]
DISPATCHER = REPO / "modules/agents/dispatcher.py"

DATASETS = [
    {"name": "agent-gen-1789603107", "count": 60, "source": "training"},
    {"name": "dataset_x", "count": 5, "source": "generated"},
]


class TestTheIncident:
    @pytest.mark.parametrize(
        "message",
        [
            "I want to train a 1B model on the agent-gen-1789603107 dataset.",
            "Go ahead and train on agent-gen-1789603107",
            "Start training with the agent-gen-1789603107 dataset",
            "please use agent-gen-1789603107",
        ],
    )
    def test_found_anywhere_in_the_sentence(self, message):
        """The name usually sits mid-sentence, which a first-token rule cannot catch."""
        assert select_training_dataset(message, DATASETS) == "agent-gen-1789603107"

    @pytest.mark.parametrize("message", ["Go", "Go ahead", "Start training", "train it"])
    def test_first_token_is_not_a_dataset(self, message):
        assert select_training_dataset(message, DATASETS) is None


class TestQuoted:
    def test_quoted_name_is_used_when_real(self):
        assert select_training_dataset('train "dataset_x" now', DATASETS) == "dataset_x"

    def test_quoted_name_is_dropped_when_not_real(self):
        """The old version never checked, so any string became a dataset name."""
        assert select_training_dataset('train "nope" now', DATASETS) is None

    def test_a_real_name_in_the_body_beats_a_fake_quote(self):
        msg = 'train "nope" using agent-gen-1789603107'
        assert select_training_dataset(msg, DATASETS) == "agent-gen-1789603107"


class TestAmbiguityAndEdges:
    def test_two_mentions_pick_nothing(self):
        msg = "compare agent-gen-1789603107 and dataset_x"
        assert select_training_dataset(msg, DATASETS) is None

    def test_prefix_overlap_resolves_to_the_longer_name(self):
        ds = [{"name": "ds_a"}, {"name": "ds_a_twist"}]
        assert select_training_dataset("train on ds_a_twist", ds) == "ds_a_twist"

    @pytest.mark.parametrize("message", ["", None])
    def test_empty_message(self, message):
        assert select_training_dataset(message or "", DATASETS) is None

    @pytest.mark.parametrize("datasets", [[], None, [{}], [{"name": ""}], ["x"], [None]])
    def test_malformed_list_does_not_raise(self, datasets):
        assert select_training_dataset("train agent-gen-1789603107", datasets or []) is None

    def test_name_extraction(self):
        assert dataset_names(DATASETS) == ["agent-gen-1789603107", "dataset_x"]
        assert dataset_names([{"name": " x "}, {"nope": 1}]) == ["x"]


class TestSingleSource:
    """Reimplementing the matcher here would give three different answers."""

    def test_reuses_the_existing_matcher(self):
        src = (REPO / "server/core/train_target.py").read_text(encoding="utf-8")
        assert "from server.core.auto_intent import find_folder_in_text" in src

    def test_same_input_same_answer(self):
        from server.core.auto_intent import find_folder_in_text

        msg = "train on agent-gen-1789603107"
        names = dataset_names(DATASETS)
        assert select_training_dataset(msg, DATASETS) == find_folder_in_text(msg, names)


class TestWiring:
    @pytest.fixture(scope="class")
    def src(self):
        return DISPATCHER.read_text(encoding="utf-8")

    def test_caller_uses_the_resolver(self, src):
        assert "_ds_all = self._list_training_datasets()" in src
        assert "resolve_training_dataset(message, _ds_all)" in src

    def test_first_token_regex_is_gone(self, src):
        """That one line is what turned `Go` into a dataset."""
        assert "^([a-zA-Z0-9][a-zA-Z0-9_-]+)" not in src

    def test_listing_has_a_single_source(self, src):
        """Resolution and context assembly must read the same list, or a chosen
        name can silently be missing from it."""
        assert "def _list_training_datasets(self)" in src
        assert src.count("def _list_training_datasets") == 1
        assert "datasets = self._list_training_datasets()" in src

    def test_withholding_the_card_is_logged(self, src):
        """No card means the user sees a question instead of a run, so the
        reason has to be traceable afterwards."""
        tree = ast.parse(src)
        warns = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("logger.warning")
        ]
        assert any("confirm card withheld" in ast.unparse(w) for w in warns)

    def test_a_name_outside_the_list_is_not_passed_on(self, src):
        """Passing a name like `Go` makes the lookup miss and report 0 samples."""
        assert '_ds_hint["job_name"] = _ds_pick' in src
        assert "if _ds_pick:" in src

    def test_the_model_named_in_the_message_is_passed_on(self, src):
        """Without it the KBD model from an earlier run keeps winning."""
        assert "resolve_requested_model(message, self._installed_model_names())" in src
        assert '_ds_hint["requested_model"] = _model_pick' in src


# ── The base model named in the message ─────────────────────────────────────
# People name a model by size or family ("the 0.5B", "Qwen"), not by its repo
# id. Without resolving that, the KBD model from an earlier run keeps winning.

MODELS = ["HuggingFaceTB--SmolLM2-135M-Instruct", "Qwen--Qwen2.5-0.5B-Instruct"]


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Train 135M llm with dataset_20260922_204419", MODELS[0]),
        ("Train 0.5B llm with dataset_20260922_204419", MODELS[1]),
        ("Train 0.5 B llm with ds", MODELS[1]),
        ("train with Qwen", MODELS[1]),
        ("use SmolLM2 please", MODELS[0]),
        ("Train HuggingFaceTB--SmolLM2-135M-Instruct on ds", MODELS[0]),
    ],
)
def test_a_named_model_is_resolved(message, expected):
    assert resolve_requested_model(message, MODELS) == expected


@pytest.mark.parametrize(
    "message",
    [
        "just start training",
        "go",
        "compare 135M and 0.5B",  # two named: a guess would pick the wrong one
        "train the instruct model",  # shared by every name, so it identifies none
        "",
    ],
)
def test_nothing_is_invented(message):
    assert resolve_requested_model(message, MODELS) is None


def test_an_empty_install_matches_nothing():
    assert resolve_requested_model("train 0.5B", []) is None


def test_handles_skip_pieces_every_model_shares():
    handles = model_handles("Qwen--Qwen2.5-0.5B-Instruct")
    assert "0.5b" in handles and "qwen2.5" in handles
    assert "instruct" not in handles
