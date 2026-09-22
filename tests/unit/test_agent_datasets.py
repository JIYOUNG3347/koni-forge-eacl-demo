"""{"datasets": [{"name": "agent-gen-1786441731_twist", "sample_count": 40},
                  {"name": "agent-gen-1786670015_twist", "sample_count": 172},
                  {"name": "dataset_20260615_105303", "sample_count": 500}, …]}

A dataset that was just created was missing, because the tool looked at one root and one file::

    dataset_dir = Path("/storage/corpus")     # never sees generated_corpus
    qa_file = d / "qa_dataset.json"           # never sees qa_dataset_preprocessed.json

A run that skipped augmentation matched neither condition and vanished, so the
LLM picked a `_twist` dataset from an older run instead.
"""

from pathlib import Path

import pytest

from server.core.agent_datasets import DATASET_ROOTS, listing_hint, order_datasets
from server.core.corpus_files import REFINED_SUFFIXES, select_data_files

_ROOT = Path(__file__).resolve().parents[2]

INCIDENT = [
    {"name": "agent-gen-1786441731_twist", "sample_count": 40},
    {"name": "agent-gen-1786670015_twist", "sample_count": 172},
    {"name": "agent-gen-1788502867", "sample_count": 50},
]


# ── List order and guidance ─────────────────────────────────────────────────


def test_pipeline_dataset_comes_first():
    out = order_datasets(INCIDENT, "agent-gen-1788502867")
    assert out[0]["name"] == "agent-gen-1788502867"
    assert out[0]["requested"] is True


def test_unrequested_entries_keep_their_order():
    out = order_datasets(INCIDENT, "agent-gen-1788502867")
    assert [d["name"] for d in out[1:]] == [d["name"] for d in INCIDENT[:2]]


def test_no_request_leaves_the_list_alone():
    for req in (None, "", "   "):
        assert [d["name"] for d in order_datasets(INCIDENT, req)] == [d["name"] for d in INCIDENT]


def test_unknown_request_does_not_invent_anything():
    """Leading with a stale or misspelled name starts training on a dataset that does not exist."""
    out = order_datasets(INCIDENT, "does-not-exist")
    assert [d["name"] for d in out] == [d["name"] for d in INCIDENT]
    assert listing_hint(out, "does-not-exist") == ""


def test_hint_names_the_dataset_and_forbids_picking():
    hint = listing_hint(order_datasets(INCIDENT, "agent-gen-1788502867"), "agent-gen-1788502867", "en")
    assert "agent-gen-1788502867" in hint
    assert "Do not pick" in hint


def test_the_single_dataset_root_is_searched():
    """This used to be `test_both_roots_are_searched`. Both roots were needed
    because promotion left copies in two places, and promotion itself is gone.

    """
    assert any(root.endswith("/corpus") for root in DATASET_ROOTS)
    assert not any(root.endswith("/generated_corpus") for root in DATASET_ROOTS)


# ── File selection rule ─────────────────────────────────────────────────────


def test_preprocessed_replaces_the_original():
    """Including the original revives filtered records and trains the survivors twice."""
    files = [
        "generation_params.json",
        "preprocessing_stats.txt",
        "provenance.json",
        "qa_dataset.json",
        "qa_dataset_preprocessed.json",
    ]
    assert select_data_files(files) == ["qa_dataset_preprocessed.json"]


def test_refined_wins_over_the_original():
    assert select_data_files(["qa_dataset.json", "qa_dataset_refined.json"]) == ["qa_dataset_refined.json"]


def test_korean_twist_folder_is_unaffected():
    """A training folder with one file is unaffected by this change."""
    assert select_data_files(["qa_dataset.json"]) == ["qa_dataset.json"]


def test_every_suffix_is_declared_in_one_place():
    assert set(REFINED_SUFFIXES) == {"_twist", "_preprocessed", "_refined"}


def test_stats_sidecars_are_still_excluded():
    assert select_data_files(["qa_dataset_refined.json", "qa_dataset_refined_stats.json"]) == [
        "qa_dataset_refined.json"
    ]


# ── Wiring guard ────────────────────────────────────────────────────────────

_SPEC = (_ROOT / "modules" / "agents" / "specialists" / "tuning_specialist.py").read_text(encoding="utf-8")


def test_tool_searches_both_roots_and_uses_the_shared_file_rule():
    assert "DATASET_ROOTS" in _SPEC
    assert "select_data_files(" in _SPEC
    # The old one-root, one-file code must not still be here.
    assert 'dataset_dir = Path("/storage/corpus")' not in _SPEC
    assert 'qa_file = d / "qa_dataset.json"' not in _SPEC


def test_tool_puts_the_pipeline_dataset_first():
    assert "order_datasets(" in _SPEC
    assert '_pipeline_identity_hint("dataset")' in _SPEC


def test_tool_states_the_instruction_only_when_it_applies():
    """Saying "use this" for a name that is not in the list dies with a 404."""
    assert "listing_hint(" in _SPEC
    block = _SPEC[_SPEC.index("hint = listing_hint(") :][:260]
    assert "if hint:" in block


# ── Nothing to train on ─────────────────────────────────────────────────────
# This edition has no QA generation, so a dataset arrives only by upload. The
# listing must say that rather than leave the LLM to invent a name.


def test_an_empty_listing_asks_for_an_upload():
    hint = listing_hint([])
    assert "upload" in hint.lower()
    assert "start_training_job" in hint


def test_folders_without_data_ask_for_an_upload_too():
    hint = listing_hint([{"name": "d1", "has_data": False}])
    assert "upload" in hint.lower()


def test_a_usable_dataset_needs_no_hint():
    assert listing_hint([{"name": "d1", "has_data": True}]) == ""


def test_a_pipeline_choice_still_wins():
    hint = listing_hint([{"name": "d1", "has_data": True}], requested="d1")
    assert "d1" in hint and "Do not pick a different dataset" in hint


# ── No arbitrary training target ────────────────────────────────────────────
# KBD on one document says nothing about an unrelated dataset that happens to
# sit first in the list. Pre-filling the confirm card with one is how the wrong
# data gets trained.

DISPATCHER = Path(__file__).resolve().parents[2] / "modules" / "agents" / "dispatcher.py"


def test_the_training_context_picks_no_dataset_by_position():
    src = DISPATCHER.read_text(encoding="utf-8")
    assert 'ds_name = datasets[0]["name"]' not in src


def test_both_answers_come_from_one_place():
    """Two paths open the confirm card; a guard on only one leaves the other."""
    src = DISPATCHER.read_text(encoding="utf-8")
    assert src.count("def _unresolved_dataset_message") == 1
    assert src.count("self._unresolved_dataset_message(ctx)") == 2
    body = src[src.index("def _unresolved_dataset_message"):]
    body = body[: body.index("async def ")]
    assert "train_dataset_unresolved" in body, "the known-datasets case must be named"
    assert "res_no_training_dataset" in body, "the empty case must still ask for an upload"


@pytest.mark.parametrize(
    "anchor",
    ['_p_dataset = ctx.get("dataset_name", "")', 'dataset = ctx.get("dataset_name", "")'],
)
def test_each_card_path_stops_when_no_dataset_is_resolved(anchor):
    src = DISPATCHER.read_text(encoding="utf-8")
    block = src[src.index(anchor):][:900]
    assert "_unresolved_dataset_message" in block
    assert "return" in block, "it must stop rather than fall through to the card"


def test_the_unresolved_text_lists_the_options_and_refuses_to_start():
    from server.core.agent_runtime_texts import text

    msg = text("train_dataset_unresolved", "en", preview="- ds1\n- ds2")
    assert "ds1" in msg and "ds2" in msg
    assert "Training was not started." in msg
