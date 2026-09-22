"""Adjust training parameters when the dataset is small.

The arithmetic is unchanged (pinned by golden tests). The result is an
:class:`Adjustment` saying what changed and why, so the caller can show it.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple

#: Minimum optimizer steps for training to mean anything. The basis of the adjustment.
MIN_STEPS = 30

#: Floor the adjustment will not reduce the batch below.
MIN_BATCH = 2

#: Floor for the epoch count when the adjustment raises it.
MIN_EPOCHS = 5

REASON_FEW_STEPS = "steps_below_min"
REASON_BATCH_LOWERED = "batch_lowered"
REASON_EPOCHS_RAISED = "epochs_raised"


class Adjustment(NamedTuple):
    """Adjustment result. With ``adjusted`` False the rest is the request verbatim."""

    epochs: int
    batch_size: int
    info: Dict[str, Any]


def plan(
    sample_count: int,
    epochs: int,
    batch_size: int,
) -> Adjustment:
    """Requested parameters plus a sample count to the values actually used.

    ``sample_count <= 0`` (unknown) adjusts nothing — changing values from an
    unknown would be another silent malfunction.

    """
    before = {"epochs": epochs, "batch_size": batch_size}
    estimated_steps = (sample_count // max(batch_size, 1)) * epochs if sample_count > 0 else 0

    if not (estimated_steps > 0 and estimated_steps < MIN_STEPS):
        return Adjustment(
            epochs,
            batch_size,
            {
                "adjusted": False,
                "sample_count": sample_count,
                "estimated_steps": estimated_steps,
                "min_steps": MIN_STEPS,
                "before": before,
                "after": dict(before),
                "reasons": [],
            },
        )

    reasons: List[str] = [REASON_FEW_STEPS]

    if batch_size > MIN_BATCH and sample_count <= batch_size * 4:
        batch_size = max(1, min(MIN_BATCH, sample_count // 2))
        reasons.append(REASON_BATCH_LOWERED)

    steps_per_epoch = max(sample_count // max(batch_size, 1), 1)
    min_epochs = max((MIN_STEPS // steps_per_epoch) + 1, MIN_EPOCHS)
    if min_epochs > epochs:
        reasons.append(REASON_EPOCHS_RAISED)
    epochs = max(epochs, min_epochs)

    new_steps = steps_per_epoch * epochs

    after = {"epochs": epochs, "batch_size": batch_size}
    return Adjustment(
        epochs,
        batch_size,
        {
            "adjusted": after != before,
            "sample_count": sample_count,
            "estimated_steps": estimated_steps,
            "planned_steps": new_steps,
            "min_steps": MIN_STEPS,
            "before": before,
            "after": after,
            "reasons": reasons,
        },
    )


def changed_fields(info: Dict[str, Any]) -> Dict[str, Any]:
    """Only the fields that actually changed, as ``{key: (before, after)}``.

    Reporting an unchanged field as changed overstates what the adjustment did.
    """
    before = info.get("before") or {}
    after = info.get("after") or {}
    return {k: (before.get(k), after.get(k)) for k in before if before.get(k) != after.get(k)}


__all__ = [
    "MIN_BATCH",
    "MIN_EPOCHS",
    "MIN_STEPS",
    "REASON_BATCH_LOWERED",
    "REASON_EPOCHS_RAISED",
    "REASON_FEW_STEPS",
    "Adjustment",
    "changed_fields",
    "plan",
]
