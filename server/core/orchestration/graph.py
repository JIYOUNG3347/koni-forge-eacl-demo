"""The pipeline as a conditional edge graph with guards.

A pure formalisation of the fixed chain map (``dispatcher.PIPELINE_STAGES``)

and the dynamic KBD branch (``dispatcher._resolve_kbd_path``). The order is a
1:1 port, so behaviour is preserved.

Concepts:
  * **Node**: a stage-completion event (src is the finished stage) and the
    specialist to run next (dst is the agent). A dst of ``None`` is terminal.
  * **Edge guard**: takes ``(Blackboard, meta)`` and decides whether the
    transition is valid. ``Blackboard`` is state persisted in the session;
    ``meta`` is the runtime signal for this transition — ``has_training_data``,
    for example, is never persisted and only arrives as chain metadata.

Careful: a stage name is not an agent name. The keys of PIPELINE_STAGES are
*finished stages* and next_agent is the *agent to run*: the finished
"retrieval" stage leads to the "boundary" agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from server.core.orchestration.blackboard import _FT_KEYWORDS, Blackboard

# Guard signature: (persisted state, runtime metadata for this transition) -> bool
GuardFn = Callable[[Blackboard, dict], bool]


def _always(_bb: Blackboard, _meta: dict) -> bool:
    return True


def _needs_ft(bb: Blackboard, meta: dict) -> bool:
    """Whether KBD recommended fine-tuning or hybrid.

    Runtime metadata wins: right after KBD finishes the recommendation only
    exists there. Otherwise fall back to the kbd stage already recorded on the
    blackboard.
    """
    rec = str(meta.get("recommendation", "")).lower()
    if rec:
        return any(kw in rec for kw in _FT_KEYWORDS)
    return bb.needs_finetuning()


@dataclass(frozen=True)
class Edge:
    """Conditional edge: src stage finished, so run dst when the guard passes.

    Attributes:
        src: the finished stage name (retrieval, kbd, tuning).
        dst: the agent to run next; None is terminal.
        guard: (Blackboard, meta) -> bool, whether this transition is valid.
        announce_key: key for the announcement text, filled in by the dispatcher.
        inline_continue: run straight on without a confirmation card, even in guided.
    """

    src: str
    dst: Optional[str]
    guard: GuardFn = _always
    announce_key: str = ""
    inline_continue: bool = False


@dataclass
class Graph:
    """A set of conditional edges plus the next-node decision."""

    edges: list[Edge] = field(default_factory=list)

    def candidates(self, bb: Blackboard, src: str, meta: dict | None = None) -> list[Edge]:
        """Edges out of src whose guard passes, in definition order.

        The supervisor may only choose from this list.
        """
        meta = meta or {}
        return [e for e in self.edges if e.src == src and e.guard(bb, meta)]

    def next_edge(self, bb: Blackboard, src: str, meta: dict | None = None) -> Optional[Edge]:
        """First passing edge out of src — the same deterministic choice as legacy.

        Used as the default when the supervisor is not consulted.
        """
        cands = self.candidates(bb, src, meta)
        return cands[0] if cands else None

    def sources(self) -> set[str]:
        return {e.src for e in self.edges}


def edge_is_wireable(edge: Optional[Edge], known_agents) -> bool:
    """Whether a transition edge can actually be wired.

    Same principle as ``resolve_route``: the graph decision is only used when it
    cross-checks against a registered agent, otherwise fall back to legacy.

    Args:
        edge: the ``Graph.next_edge`` result, possibly None for an unknown src.
        known_agents: agents actually registered on the dispatcher.

    Returns:
        True to wire this edge. A terminal edge (dst=None) is always wireable.
    """
    if edge is None:
        return False
    return edge.dst is None or edge.dst in set(known_agents)


def default_pipeline_graph() -> Graph:
    """retrieval -> KBD -> (training | end)."""
    return Graph(
        edges=[
            Edge("retrieval", "boundary", announce_key="retrieval_to_kbd"),
            Edge("kbd", "tuning", guard=_needs_ft, announce_key="kbd_to_tuning"),
            Edge("kbd", None, guard=lambda bb, m: not _needs_ft(bb, m), announce_key="kbd_rag_sufficient"),
            Edge("tuning", None, announce_key="tuning_terminal"),
        ]
    )
