"""Build the LLM prompt for a whole workflow plan and parse the response.

Beyond the supervisor's "pick the next node", this composes the entire

Safety rules, same lineage as the supervisor:
  * The parser returns only plans that pass both :func:`plan.parse_plan_dict`
    (shape) and :func:`plan.validate_plan` (meaning), so an unregistered node
    or guard can never reach execution.
  * Any failure returns None and the caller falls back to the static graph.
  * No theatre: the prompt says "only the nodes you need". Suppressing idle

This is a separate module rather than part of supervisor.py because plan.py
imports supervisor (END), so a supervisor-to-plan import would be circular.
The planner sits downstream of both.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

from server.core.orchestration.blackboard import Blackboard
from server.core.orchestration.lang_pack import guard_descriptions as _guard_descriptions
from server.core.orchestration.lang_pack import route_hints as _route_hints
from server.core.orchestration.lang_pack import texts as _texts
from server.core.orchestration.plan import (
    Plan,
    default_pipeline_plan,
    parse_plan_dict,
    plan_to_dict,
    validate_plan,
)
from server.core.orchestration.supervisor import _state_summary


def build_plan_prompt(bb: Blackboard, goal: str, history: str = "", lang: Optional[str] = None) -> str:
    """Prompt that asks the LLM for a plan.

    Args:
        bb: current shared state (completed stages and the KBD verdict).
        goal: the user's goal, verbatim.
        history: summary of the previous run's traverse (``summarize_traverse``).
            An empty string omits the section entirely.
    """
    t = _texts(lang)
    hints = _route_hints(lang)
    guards = _guard_descriptions(lang)
    node_lines = "\n".join(f'  - "{n}": {h}' for n, h in hints.items())
    guard_lines = "\n".join(f'  - "{g}": {d}' for g, d in guards.items())
    example = json.dumps(plan_to_dict(default_pipeline_plan()), ensure_ascii=False)
    history_section = t["section_history"].format(value=history) if history else ""
    return (
        t["plan_role"]
        + t["section_objective"].format(value=goal)
        + f"{t['section_state']}{_state_summary(bb, lang)}\n"
        + history_section
        + t["section_nodes"]
        + f"{node_lines}\n"
        + t["section_guards"]
        + f"{guard_lines}\n"
        + t["section_rules"]
        + t["plan_rules"].format()
        + t["section_example"]
        + example
    )


def _iter_json_candidates(text: str) -> Iterator[str]:
    """Candidate JSON object strings from an LLM response, best first.

    (1) the whole text, (2) inside a code fence, (3) the top-level ``{...}``
    found by a brace scan that understands string literals.
    """
    stripped = text.strip()
    if stripped:
        yield stripped

    # Inside a code fence (```json ... ``` or ``` ... ```)
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith(("json", "JSON")):
            block = block[4:]
        block = block.strip()
        if block:
            yield block

    # Balanced-brace scan, ignoring braces and escapes inside JSON strings.
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def parse_plan_response(llm_text: str) -> Optional[Plan]:
    """Planner response to a validated :class:`Plan`, or None on failure.

    Returns the first candidate that passes both shape and meaning checks, so
    the caller can feed the result straight into ``plan_to_graph``.
    """
    if not llm_text:
        return None
    for candidate in _iter_json_candidates(llm_text):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        plan = parse_plan_dict(data)
        if plan is not None and not validate_plan(plan):
            return plan
    return None
