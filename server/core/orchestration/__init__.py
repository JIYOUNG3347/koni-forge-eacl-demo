"""Pure modules with no fastapi, celery or torch dependency.

Holds the data structures and decision logic that replace the keyword
dispatcher and its fixed chain map with an LLM-supervised dynamic graph.
"""

from server.core.orchestration.blackboard import (
    Blackboard,
    StageRecord,
    from_session,
)
from server.core.orchestration.graph import (
    Edge,
    Graph,
    default_pipeline_graph,
    edge_is_wireable,
)
from server.core.orchestration.plan import (
    AGENT_COMPLETION_STAGE,
    GUARD_DESCRIPTIONS,
    GUARD_REGISTRY,
    Plan,
    PlanEdge,
    default_pipeline_plan,
    parse_plan_dict,
    plan_to_dict,
    plan_to_graph,
    validate_plan,
)
from server.core.orchestration.planner import (
    build_plan_prompt,
    parse_plan_response,
)
from server.core.orchestration.router import (
    SPECIALIST_NODES,
    fallback_route,
    initial_route_candidates,
    is_auto_planner,
    is_graph_mode,
    orchestrator_mode,
    planner_mode,
    resolve_route,
)
from server.core.orchestration.supervisor import (
    END,
    build_planner_prompt,
    parse_planner_decision,
)
from server.core.orchestration.trace import (
    TRACE_CAP,
    TRACE_KEY,
    append_trace,
    execute_event,
    plan_event,
    route_event,
    summarize_traverse,
    transition_event,
)

__all__ = [
    "Blackboard",
    "StageRecord",
    "from_session",
    "Edge",
    "Graph",
    "default_pipeline_graph",
    "edge_is_wireable",
    "END",
    "build_planner_prompt",
    "parse_planner_decision",
    "SPECIALIST_NODES",
    "fallback_route",
    "initial_route_candidates",
    "is_auto_planner",
    "is_graph_mode",
    "orchestrator_mode",
    "planner_mode",
    "resolve_route",
    "TRACE_CAP",
    "TRACE_KEY",
    "append_trace",
    "execute_event",
    "plan_event",
    "route_event",
    "summarize_traverse",
    "transition_event",
    "AGENT_COMPLETION_STAGE",
    "GUARD_DESCRIPTIONS",
    "GUARD_REGISTRY",
    "Plan",
    "PlanEdge",
    "build_plan_prompt",
    "default_pipeline_plan",
    "parse_plan_dict",
    "parse_plan_response",
    "plan_to_dict",
    "plan_to_graph",
    "validate_plan",
]
