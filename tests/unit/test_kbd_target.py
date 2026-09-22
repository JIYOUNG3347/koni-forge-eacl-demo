"""The list was sorted by chunk count descending only, so the LLM always picked
the largest. Policy: the user's choice wins, and size only breaks ties.
"""

from pathlib import Path

from server.core.kbd_target import select_kbd_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DOCS = [
    {"id": "pipeline_docs", "name": "pipeline_docs", "chunks": 4075},
    {"id": "dataset_20260623_021922", "name": "dataset_20260623_021922", "chunks": 848},
    {"id": "dataset_20260617_075943", "name": "dataset_20260617_075943", "chunks": 89},
]


def test_user_specified_dataset_wins_over_largest():
    docs, requested = select_kbd_target(_DOCS, "run KBD on dataset_20260623_021922")
    assert requested == "dataset_20260623_021922"
    assert docs[0]["id"] == "dataset_20260623_021922"
    assert docs[0].get("requested") is True


def test_no_specification_keeps_largest_first():
    # Without a request, the previous behaviour holds (largest first).
    docs, requested = select_kbd_target(_DOCS, "run the KBD analysis")
    assert requested is None
    assert docs[0]["id"] == "pipeline_docs"
    assert [d["chunks"] for d in docs] == [4075, 848, 89]


def test_remaining_docs_still_sorted_by_size():
    docs, _ = select_kbd_target(_DOCS, "use dataset_20260617_075943")
    assert docs[0]["id"] == "dataset_20260617_075943"
    # The rest keep their size order.
    assert [d["chunks"] for d in docs[1:]] == [4075, 848]


def test_ambiguous_mention_falls_back_to_size():
    # Two names at once cannot be resolved, so the previous rule applies.
    docs, requested = select_kbd_target(_DOCS, "which is better, dataset_20260623_021922 or dataset_20260617_075943?")
    assert requested is None
    assert docs[0]["id"] == "pipeline_docs"


def test_unknown_name_in_text_is_ignored():
    # A name that does not exist is never adopted (no hallucination).
    _, requested = select_kbd_target(_DOCS, "run KBD on ghost_dataset")
    assert requested is None


def test_empty_inputs_are_safe():
    assert select_kbd_target([], "anything") == ([], None)
    docs, requested = select_kbd_target(_DOCS, "")
    assert requested is None and docs[0]["id"] == "pipeline_docs"


def test_input_documents_not_mutated():
    original = [dict(d) for d in _DOCS]
    select_kbd_target(_DOCS, "dataset_20260623_021922")
    assert _DOCS == original  # the input is unchanged (no requested key leaks)


def test_boundary_specialist_wired_and_old_sort_removed():
    src = (PROJECT_ROOT / "modules/agents/specialists/boundary_specialist.py").read_text(encoding="utf-8")
    assert "select_kbd_target(docs, self._recent_user_text())" in src
    assert 'docs.sort(key=lambda d: d["chunks"], reverse=True)' not in src
