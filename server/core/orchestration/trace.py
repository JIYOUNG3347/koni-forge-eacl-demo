"""Traverse events recorded on the session, three kinds:

* ``route``      — initial routing: whether the supervisor decision was adopted
                   or it fell back to legacy keywords.
* ``transition`` — stage finished, next agent decided (at proposal time). In
                   graph mode the edge's ``announce_key`` is recorded too.
* ``execute``    — the transition ran (auto, or after guided approval).

"""

from __future__ import annotations

import time
from typing import Optional

TRACE_KEY = "traverse"
TRACE_CAP = 200  # events per session; the oldest are dropped beyond this


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def route_event(
    *,
    mode: str,
    decision: str,
    planner_decision: Optional[str] = None,
    legacy_decision: str = "",
    message_preview: str = "",
) -> dict:
    """Initial routing decision.

    Args:
        mode: the orchestrator_mode() result ("legacy" or "graph").
        decision: the agent finally adopted.
        planner_decision: the supervisor's parsed answer (None when not called or failed).
        legacy_decision: the route_match fallback result.
        message_preview: first 80 characters of the user message, for debugging.
    """
    return {
        "kind": "route",
        "at": _now(),
        "mode": mode,
        "decision": decision,
        "used": "planner" if (planner_decision and decision == planner_decision) else "legacy",
        "planner_decision": planner_decision,
        "legacy_decision": legacy_decision,
        "message_preview": message_preview[:80],
    }


def transition_event(
    *,
    src: str,
    dst: Optional[str],
    mode: str,
    announce_key: str = "",
) -> dict:
    """Transition decision, recorded at proposal time; execution has its own event."""
    return {
        "kind": "transition",
        "at": _now(),
        "mode": mode,  # "graph" | "legacy"
        "src": src,
        "dst": dst,
        "announce_key": announce_key,
        "terminal": dst is None,
    }


def execute_event(*, agent: str, src: str = "") -> dict:
    """Transition execution."""
    return {"kind": "execute", "at": _now(), "agent": agent, "src": src}


def plan_event(*, goal: str, source: str, edge_count: int = 0) -> dict:
    """Workflow plan adoption.

    Args:
        goal: the plan goal, verbatim.
        source: "planner" when the LLM plan was adopted, "static_fallback" when
            generation or validation failed, or planning was disabled.
        edge_count: edges in the adopted plan (0 on fallback).
    """
    return {
        "kind": "plan",
        "at": _now(),
        "goal": goal[:120],
        "source": source,
        "edge_count": edge_count,
    }


def summarize_traverse(events: "Optional[list]", max_lines: int = 6) -> str:
    """Short summary of the previous traverse, for the planner prompt. Empty when absent."""
    if not events:
        return ""
    lines: list[str] = []

    plans = [e for e in events if isinstance(e, dict) and e.get("kind") == "plan"]
    if plans:
        last = plans[-1]
        src = "LLM plan adopted" if last.get("source") == "planner" else "static graph (plan fallback)"
        edges = last.get("edge_count")
        lines.append(f"- previous plan: {src}" + (f" ({edges} edges)" if edges else ""))

    executed = [
        str(e.get("agent")) for e in events if isinstance(e, dict) and e.get("kind") == "execute" and e.get("agent")
    ]
    if executed:
        lines.append("- executed path: " + " -> ".join(executed[:10]))

    keys = [str(e.get("announce_key", "")) for e in events if isinstance(e, dict)]
    if any(k.startswith("reflection:replan") for k in keys):
        lines.append("- the previous run proposed or ran retraining after a low evaluation score — include a training stage")
    elif any(k == "reflection:suppressed" for k in keys):
        lines.append("- the previous run detected a low evaluation score (automatic retraining was suppressed)")

    routes = [e for e in events if isinstance(e, dict) and e.get("kind") == "route"]
    fallbacks = sum(1 for e in routes if e.get("mode") == "graph" and e.get("used") == "legacy")
    if fallbacks:
        lines.append(f"- supervisor routing fell back {fallbacks} time(s) (parse failure or ambiguity)")

    return "\n".join(lines[:max_lines])


def append_trace(session: dict, event: dict, cap: int = TRACE_CAP) -> None:
    """Append a traverse event to the session, dropping the oldest beyond the cap.

    A session that is not a dict is silently ignored — recording must never kill execution.
    """
    if not isinstance(session, dict):
        return
    trace = session.setdefault(TRACE_KEY, [])
    trace.append(event)
    if len(trace) > cap:
        del trace[: len(trace) - cap]
