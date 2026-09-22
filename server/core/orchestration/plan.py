"""Workflow plan — a declarative graph the supervisor can compose.

The LLM cannot produce Python callables, so guards are a fixed registry of
named predicates and the model only combines names. Unknown nodes or guards
are rejected by :func:`validate_plan`, so a hallucinated plan never reaches
execution.

Plan JSON (what the planner LLM emits — vocabulary is agent names only)::

    {"goal": "...", "edges": [
        {"src": "retrieval", "dst": "boundary"},
        {"src": "boundary", "dst": "tuning", "guard": "needs_ft"},
        {"src": "boundary", "dst": "__end__", "guard": "rag_sufficient"}
    ]}

``plan_to_graph`` translates each ``src`` (agent name) into the stage name
that agent emits on completion (``AGENT_COMPLETION_STAGE``) so the existing
:class:`Graph` executor runs the plan unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from server.core.orchestration.blackboard import Blackboard
from server.core.orchestration.graph import (
    Edge,
    Graph,
    GuardFn,
    _always,
    _needs_ft,
)
from server.core.orchestration.router import SPECIALIST_NODES
from server.core.orchestration.supervisor import END

# agent name -> stage name it emits on completion (chain_signal vocabulary)
AGENT_COMPLETION_STAGE: dict[str, str] = {
    "retrieval": "retrieval",
    "boundary": "kbd",
    "tuning": "tuning",
}


def _rag_sufficient(bb: Blackboard, meta: dict) -> bool:
    return not _needs_ft(bb, meta)


GUARD_REGISTRY: dict[str, GuardFn] = {
    "always": _always,
    "needs_ft": _needs_ft,
    "rag_sufficient": _rag_sufficient,
}

# One-line guard descriptions for the planner prompt. Keys must match
# GUARD_REGISTRY 1:1 (enforced by tests).
GUARD_DESCRIPTIONS: dict[str, str] = {
    "always": "Unconditional transition (same as omitting the guard)",
    "needs_ft": "KBD recommended Fine-Tuning or Hybrid",
    "rag_sufficient": "KBD verdict: retrieval alone is sufficient (no fine-tuning)",
}


@dataclass(frozen=True)
class PlanEdge:
    """One conditional edge of a plan, in agent-name space.

    Attributes:
        src: agent whose completion triggers this edge (SPECIALIST_NODES).
        dst: agent to run next. None means terminal (END).
        guard: a GUARD_REGISTRY name. Defaults to "always".
    """

    src: str
    dst: Optional[str]
    guard: str = "always"


@dataclass(frozen=True)
class Plan:
    """A workflow plan composed by the supervisor (immutable)."""

    goal: str = ""
    edges: "tuple[PlanEdge, ...]" = ()


def _normalize_dst(raw: Any) -> "tuple[Optional[str], bool]":
    """Raw dst value -> (normalized, valid). END sentinel / null -> None."""
    if raw is None or raw == END or raw == "":
        return None, True
    if isinstance(raw, str):
        return raw, True
    return None, False


def parse_plan_dict(data: Any) -> Optional[Plan]:
    """Plan dict (parsed JSON) -> :class:`Plan`, or None on a shape error.

    Only checks shape (keys and types). Semantic checks (node / guard
    existence) live in :func:`validate_plan`; callers need both.
    """
    if not isinstance(data, dict):
        return None
    raw_edges = data.get("edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        return None
    edges: list[PlanEdge] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            return None
        src = item.get("src")
        if not isinstance(src, str) or not src:
            return None
        dst, ok = _normalize_dst(item.get("dst"))
        if not ok:
            return None
        guard = item.get("guard", "always")
        if not isinstance(guard, str) or not guard:
            return None
        edges.append(PlanEdge(src=src, dst=dst, guard=guard))
    goal = data.get("goal", "")
    return Plan(goal=goal if isinstance(goal, str) else "", edges=tuple(edges))


def validate_plan(plan: Plan) -> "list[str]":
    """Semantic validation — returns a list of error strings (empty = valid).

    Rules: unknown nodes / guards are rejected, duplicate (src, guard) pairs
    are rejected (first-match execution makes the later edge unreachable),
    self-loops are rejected. Cycles are allowed; chain depth is bounded by
    the dispatcher.
    """
    errors: list[str] = []
    if not plan.edges:
        return ["plan has no edges"]
    known = set(SPECIALIST_NODES)
    seen: set = set()
    for e in plan.edges:
        if e.src not in known:
            errors.append(f"unknown src node: {e.src!r}")
        if e.dst is not None and e.dst not in known:
            errors.append(f"unknown dst node: {e.dst!r}")
        if e.guard not in GUARD_REGISTRY:
            errors.append(f"unknown guard: {e.guard!r} (src={e.src})")
        if e.dst is not None and e.src == e.dst:
            errors.append(f"self-loop not allowed: {e.src} -> {e.dst}")
        key = (e.src, e.guard)
        if key in seen:
            errors.append(f"duplicate (src, guard) — unreachable edge: {key}")
        seen.add(key)
    return errors


def plan_to_graph(plan: Plan) -> Graph:
    """Validated plan -> executable :class:`Graph`.

    * src agent name -> completion stage name (AGENT_COMPLETION_STAGE).
    * announce_key is ``plan:{src}_to_{dst}`` so plan-derived transitions are
      identifiable in the traverse record.

    Only call with a plan for which :func:`validate_plan` returned [].
    """
    edges = [
        Edge(
            src=AGENT_COMPLETION_STAGE[e.src],
            dst=e.dst,
            guard=GUARD_REGISTRY[e.guard],
            announce_key=f"plan:{e.src}_to_{e.dst or 'end'}",
        )
        for e in plan.edges
    ]
    return Graph(edges=edges)


def plan_to_dict(plan: Plan) -> dict:
    """:class:`Plan` -> JSON-serialisable dict (few-shot example / provenance).

    ``parse_plan_dict(plan_to_dict(p)) == p`` round-trips.
    """
    return {
        "goal": plan.goal,
        "edges": [{"src": e.src, "dst": e.dst if e.dst is not None else END, "guard": e.guard} for e in plan.edges],
    }


def default_pipeline_plan() -> Plan:
    """Plan equivalent to ``default_pipeline_graph`` (retrieval -> KBD -> training | end)."""
    return Plan(
        goal="",
        edges=(
            PlanEdge("retrieval", "boundary"),
            PlanEdge("boundary", "tuning", "needs_ft"),
            PlanEdge("boundary", None, "rag_sufficient"),
            PlanEdge("tuning", None),
        ),
    )
