"""Prompt text for the orchestration layer (supervisor + planner).

Pure module (stdlib only). The prompt *assembly* lives in supervisor.py /
planner.py; this file holds only the wording so the two never drift.
"""

from __future__ import annotations

from typing import Dict

from server.core.lang import EN

#: One-line role of each node (specialist) — the routing evidence.
ROUTE_HINTS: Dict[str, Dict[str, str]] = {
    EN: {
        "retrieval": "RAG indexing, vector search, embeddings, document lookup",
        "boundary": "KBD knowledge-boundary measurement (what the model does and does not know, coverage, hallucination rate)",
        "tuning": "model training / fine-tuning (full fine-tuning or LoRA), training-run status",
        "__end__": "end of pipeline (no next step)",
    },
}

#: Guard (transition condition) descriptions exposed by the plan prompt.
#: Keys must match plan.GUARD_REGISTRY 1:1.
GUARD_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    EN: {
        "always": "always transition (same as omitting the guard when no branching is needed)",
        "needs_ft": "KBD recommended Fine-Tuning or Hybrid",
        "rag_sufficient": "KBD verdict: retrieval alone is sufficient (no fine-tuning needed)",
    },
}

#: State summary and prompt body wording.
TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        "none": "none",
        "completed_stages": "- Completed stages: {value}",
        "last_stage": "- Last stage: {value}",
        "kbd_recommendation": "- KBD recommendation: {rec}{coverage}",
        "coverage_suffix": ", coverage {value:.0f}%",
        "planner_role": (
            "You are the orchestrator of the KONI-Forge LLM development pipeline.\n"
            "Look at the current state and the user request, then pick **exactly one** "
            "step (node) to run next.\n"
        ),
        "section_state": "\n[Current state]\n",
        "section_goal": '\nOverall goal: "{value}"',
        "section_prev": "\n[Previous turn]\n- Agent that replied last: {agent}{tail}\n",
        "prev_tail": '\n- End of that reply: "{value}"',
        "section_request": '\n[User request]\n"{value}"\n',
        "section_candidates": "\n[Selectable nodes]\n",
        "section_rules": "\n[Rules]\n",
        "rule_explicit_intent": (
            "1. If the user explicitly asks for a specific task "
            "(e.g. 'measure coverage', 'train', 'index'), follow **that intent first**, "
            "ahead of the pipeline context that was in progress.\n"
        ),
        "rule_continuity": (
            "2. If the user message is a short reply to the previous response "
            "(especially to a question back), e.g. 'yes', 'ok', 'sure', 'go ahead', "
            "'go with that', 'do it', then **keep the agent that replied last**.\n"
        ),
        "rule_candidates_only": (
            "{n}. You must pick exactly one node name from [Selectable nodes] above. "
            "Values outside that list are forbidden.\n"
        ),
        "rule_json_only": "{n}. Answer only in the JSON format below, with no other text:\n",
        "planner_json_example": '{"next": "node name"}',
        "plan_role": (
            "You are the workflow planner of the KONI-Forge LLM development pipeline.\n"
            "Build a workflow graph (plan) in JSON that achieves the goal below.\n"
        ),
        "section_objective": '\n[Goal]\n"{value}"\n',
        "section_history": (
            "\n[Previous run record]\n{value}\n"
            "(Use this record to adjust the path — reinforce steps that went wrong "
            "and avoid unnecessary repetition.)\n"
        ),
        "section_nodes": "\n[Available nodes]\n",
        "section_guards": "\n[Available guards (transition conditions)]\n",
        "plan_rules": (
            '1. Each edge is {{"src": node, "dst": node, "guard": condition}}; when the '
            "src node finishes, the dst of the first edge whose guard holds is run.\n"
            "2. Use only the nodes needed to reach the goal. Do not add unnecessary steps.\n"
            "3. src/dst must be names from [Available nodes], and guard must be a name "
            "from [Available guards]. Omit the guard when no branching is needed.\n"
            '4. Mark termination with dst "__end__". Give the last node a terminating edge.\n'
            "5. Do not use the same guard twice on the same src. Edges where src equals "
            "dst are forbidden.\n"
            "6. Output exactly one JSON object with no other text.\n"
        ),
        "section_example": "\n[Format example — standard full pipeline]\n",
    },
}


def texts(lang: str | None = None) -> Dict[str, str]:
    return TEXTS[EN]


def route_hints(lang: str | None = None) -> Dict[str, str]:
    return ROUTE_HINTS[EN]


def guard_descriptions(lang: str | None = None) -> Dict[str, str]:
    return GUARD_DESCRIPTIONS[EN]
