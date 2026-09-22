"""Estimate a job's remaining time, saying nothing when it cannot be known.

* Running: time left in the current training stage. More rounds ahead are
  reported as ``more_work`` without estimating a total — the live state does
  not record which round is current.
* Waiting: only the job at the head of the queue gets a start estimate. Those
  behind it would have to guess how long the jobs ahead take, so they get none.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# "training - step 127/747 - loss 2.17" / "DPO training - step 3/50"
_STEP_RE = re.compile(r"step\s+(\d+)\s*/\s*(\d+)")

# Minimum progress before a rate estimate is meaningful. Estimating from one
# step bakes in the warm-up and is badly wrong.
MIN_STEPS_FOR_ETA = 3
# Below this, report "starting shortly" rather than a jittering number of seconds.
MIN_MEANINGFUL_S = 30


def parse_step(message: Optional[str]) -> Optional["tuple[int, int]"]:
    """``(step, total)`` from a progress message. None when the shape differs."""
    if not message:
        return None
    m = _STEP_RE.search(str(message))
    if not m:
        return None
    step, total = int(m.group(1)), int(m.group(2))
    if total <= 0 or step < 0 or step > total:
        return None
    return step, total


def remaining_seconds(step: int, total: int, elapsed_s: float) -> Optional[float]:
    """Seconds left in the current stage. None when the basis is too weak."""
    if total <= 0 or step < MIN_STEPS_FOR_ETA or step >= total or elapsed_s <= 0:
        return None
    per_step = elapsed_s / step
    return max(0.0, per_step * (total - step))


def running_eta(message: Optional[str], elapsed_s: float, meta: Optional[Mapping[str, Any]] = None) -> dict:
    """ETA for a running job. An empty dict when it cannot be estimated."""
    parsed = parse_step(message)
    if parsed is None:
        return {}
    remaining = remaining_seconds(parsed[0], parsed[1], elapsed_s)
    if remaining is None:
        return {}
    out: dict[str, Any] = {"eta_s": round(remaining, 1), "step": parsed[0], "step_total": parsed[1]}
    return out


def start_eta(running_etas: "list[Optional[float]]") -> Optional[float]:
    """Seconds until the head of the queue is expected to start.

    A slot frees when a running job finishes; with several running, the
    earliest to finish frees it. If any of them has an unknown ETA, nothing is
    estimated — averaging in an unknown produces a plausible but wrong number.
    """
    if not running_etas:
        return None
    known = [e for e in running_etas if e is not None]
    if not known or len(known) != len(running_etas):
        return None
    return min(known)


def format_eta(seconds: Optional[float]) -> str:
    """Human-readable form. None gives an empty string."""
    if seconds is None:
        return ""
    s = max(0, int(seconds))
    if s < MIN_MEANINGFUL_S:
        return "shortly"
    minutes = s // 60
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"about {minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"about {hours}h {minutes}m" if minutes else f"about {hours}h"
