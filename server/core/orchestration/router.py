"""Whether the dispatcher uses the supervisor or the legacy keyword router.

Pure helpers only: environment reads and the candidate list, so this is

Flag: ``ORCHESTRATOR_MODE=legacy|graph`` (legacy by default).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# Must match the specialist keys in dispatcher._agents.
SPECIALIST_NODES: list[str] = ["retrieval", "boundary", "tuning"]

_VALID_MODES = ("legacy", "graph")
_DEFAULT_MODE = "graph"


def orchestrator_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Normalised ``ORCHESTRATOR_MODE``. Unset or invalid means legacy.

    Args:
        env: environment mapping, injectable for tests. None means os.environ.
    """
    if env is None:
        import os

        env = os.environ
    raw = (env.get("ORCHESTRATOR_MODE") or "").strip().lower()
    return raw if raw in _VALID_MODES else _DEFAULT_MODE


def is_graph_mode(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether graph orchestration is active."""
    return orchestrator_mode(env) == "graph"


_VALID_PLANNER_MODES = ("static", "auto")
_DEFAULT_PLANNER_MODE = "static"


def planner_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Normalised ``PLANNER_MODE``. Unset or invalid means static.

    static = use the static graph (default_pipeline_graph) only.
    auto   = the supervisor builds a plan when an auto pipeline starts,
    """
    if env is None:
        import os

        env = os.environ
    raw = (env.get("PLANNER_MODE") or "").strip().lower()
    return raw if raw in _VALID_PLANNER_MODES else _DEFAULT_PLANNER_MODE


def is_auto_planner(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether automatic planning is active. Only meaningful in graph mode."""
    return is_graph_mode(env) and planner_mode(env) == "auto"


def initial_route_candidates() -> list[str]:
    """Candidate agents for routing a free-form user request.

    This decides which agent to send a message to, not a pipeline transition,
    so the terminal (END) is not a candidate.
    """
    return list(SPECIALIST_NODES)


def fallback_route(
    keyword_match: Optional[str],
    last_agent: Optional[str],
    known_agents: Iterable[str],
) -> str:
    """Keyword fallback routing, favouring continuity.

    Principle: **a follow-up with no new stated intent stays with the previous
    agent.** This is a deterministic backstop that does not depend on listing
    confirmation words, so any variation of "yes" reaches the previous agent

    Order: (1) keyword intent, (2) the previous agent if registered, (3) corpus.

    """
    if keyword_match:
        return keyword_match
    if last_agent and last_agent in set(known_agents):
        return last_agent
    return "retrieval"


def resolve_route(
    *,
    graph_mode: bool,
    planner_decision: Optional[str],
    known_agents: Iterable[str],
    legacy: str,
) -> str:
    """Final initial-routing decision — the logic of dispatcher._route_smart.

    In graph mode a supervisor decision naming a registered agent wins;
    otherwise (legacy mode, a failed decision, or one outside the candidates)

    Args:
        graph_mode: whether ORCHESTRATOR_MODE=graph.
        planner_decision: the parse_planner_decision result, possibly None.
        known_agents: agents actually registered, cross-checked against hallucination.
        legacy: the route_match fallback result.
    """
    if graph_mode and planner_decision and planner_decision in set(known_agents):
        return planner_decision
    return legacy
