"""Typed view of the orchestration state the dispatcher persists.

``AgentDispatcher`` keeps shared state as an untyped dict in
``pipeline_session.json``. This is a pure adapter that reads that dict and
presents it as a typed :class:`Blackboard`. It writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Keywords in a KBD recommendation that mean fine-tuning or hybrid.
# Kept identical to dispatcher._needs_finetuning so behaviour is preserved.
_FT_KEYWORDS = ("fine-tuning", "fine_tuning", "hybrid", "ft")

# Common metadata keys on a stage entry, excluded from `fields`.
_COMMON_KEYS = frozenset({"stage", "skipped", "completed_at", "skip_reason", "skipped_at"})


@dataclass(frozen=True)
class StageRecord:
    """One completed (or skipped) pipeline stage, typed.

    Attributes:
        stage: stage name (retrieval, kbd, tuning).
        completed_at: completion timestamp ("%Y-%m-%dT%H:%M:%S"), None when skipped.
        skipped: whether the stage was skipped.
        fields: the stage's extra fields (recommendation, coverage_pct, model_name, ...), verbatim.
    """

    stage: str
    completed_at: Optional[str]
    skipped: bool
    fields: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Read one of the extra fields."""
        return self.fields.get(key, default)


@dataclass
class Blackboard:
    """Shared orchestration state, as consumed by the supervisor and the guards.

    Holds a snapshot of the session dict and offers state predicates that help
    decide the next node.
    """

    session_id: str = ""
    mode: str = "guided"  # "auto" | "guided"
    stages: list[StageRecord] = field(default_factory=list)
    goal: str = ""  # the user's original goal, injected externally (not in the session)

    # ── presence and order ───────────────────────────────────
    def has(self, stage: str) -> bool:
        """Whether this stage is recorded as completed (not skipped)."""
        return any(s.stage == stage and not s.skipped for s in self.stages)

    def record(self, stage: str) -> Optional[StageRecord]:
        """The last StageRecord for this stage, or None."""
        found = [s for s in self.stages if s.stage == stage]
        return found[-1] if found else None

    def completed_stages(self) -> list[str]:
        """Completed (not skipped) stage names, in recorded order."""
        return [s.stage for s in self.stages if not s.skipped]

    def last_stage(self) -> Optional[str]:
        """The most recent completed (not skipped) stage, or None."""
        done = self.completed_stages()
        return done[-1] if done else None

    def stage_field(self, stage: str, key: str, default: Any = None) -> Any:
        """One extra field of a stage, or `default` when the stage or key is missing."""
        rec = self.record(stage)
        return rec.get(key, default) if rec else default

    # ── KBD-driven branching, the core of the conditional edges ──
    def kbd_recommendation(self) -> str:
        """The KBD recommendation text, lowercased. Empty when absent."""
        return str(self.stage_field("kbd", "recommendation", "")).lower()

    def needs_finetuning(self) -> bool:
        """Whether KBD recommended fine-tuning or hybrid.

        Same keyword test as dispatcher._needs_finetuning, so behaviour is preserved.
        """
        rec = self.kbd_recommendation()
        return any(kw in rec for kw in _FT_KEYWORDS)

    def knowledge_coverage(self) -> Optional[float]:
        """KBD knowledge coverage, in percent. None when not recorded."""
        val = self.stage_field("kbd", "coverage_pct", None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


def _to_stage_record(entry: dict) -> StageRecord:
    """Convert one session stage dict into a StageRecord."""
    extra = {k: v for k, v in entry.items() if k not in _COMMON_KEYS}
    return StageRecord(
        stage=str(entry.get("stage", "")),
        completed_at=entry.get("completed_at"),
        skipped=bool(entry.get("skipped", False)),
        fields=extra,
    )


def from_session(session: Optional[dict], goal: str = "") -> Blackboard:
    """Lossless adapter from a pipeline_session.json dict to a :class:`Blackboard`.

    Args:
        session: the dispatcher's session dict (None or empty gives an empty Blackboard).
        goal: the user's original goal, injected into the supervisor prompt.
    """
    session = session or {}
    raw_stages = session.get("stages") or []
    stages = [_to_stage_record(e) for e in raw_stages if isinstance(e, dict)]
    return Blackboard(
        session_id=str(session.get("session_id", "")),
        mode=str(session.get("mode", "guided")),
        stages=stages,
        goal=goal,
    )
