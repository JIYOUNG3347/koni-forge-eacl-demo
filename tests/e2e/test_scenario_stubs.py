"""Placeholders for the full end-to-end scenarios, which consume a GPU."""

from __future__ import annotations

import pytest

HEAVY = pytest.mark.skip(reason="placeholder — full GPU scenario, pending KONI_E2E_HEAVY")


@HEAVY
def test_guided_train_eval_roundtrip():
    """guided: train (small model, 1 epoch), evaluate, then read the results.

    Contract: the training job reaches SUCCESS with a batch_guard record, the
    checkpoint holds weights, and no evaluation score is -1.
    """


@HEAVY
def test_auto_pipeline_completes_with_graph_mode():
    """auto: ORCHESTRATOR_MODE=graph with PLANNER_MODE=auto, all six stages.

    Contract: traverse holds a plan event (source=planner or static_fallback)
    plus the stage transitions, the session records six stages, and a failure
    """


@HEAVY
def test_managed_admission_serializes_train_eval():
    """managed: submit an evaluation during training, wait, then start automatically.

    Contract: with GPU_ALLOC_MODE=managed the gpu:lease:* ledger holds the
    training lease and the evaluation task reaches SUCCESS after a retry.

    """
