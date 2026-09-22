"""Pure part of the LLM planner that replaces the keyword router.

Holds only the prompt builder and the response parser; the actual
``complete_text`` call is wired up by the dispatcher. Being pure, it unit-tests
without fastapi, celery or torch.

Role: (blackboard state + user message + valid candidate nodes) -> one node.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from server.core.orchestration.blackboard import Blackboard
from server.core.orchestration.lang_pack import route_hints as _route_hints
from server.core.orchestration.lang_pack import texts as _texts

# Sentinel candidate meaning the pipeline ends here, matching Edge.dst is None.
END = "__end__"


def _state_summary(bb: Blackboard, lang: Optional[str] = None) -> str:
    """Summarise the blackboard into a short state string for the prompt."""
    t = _texts(lang)
    completed = ", ".join(bb.completed_stages()) or t["none"]
    lines = [
        t["completed_stages"].format(value=completed),
        t["last_stage"].format(value=bb.last_stage() or t["none"]),
    ]
    rec = bb.kbd_recommendation()
    if rec:
        cov = bb.knowledge_coverage()
        cov_txt = t["coverage_suffix"].format(value=cov) if cov is not None else ""
        lines.append(t["kbd_recommendation"].format(rec=rec, coverage=cov_txt))
    return "\n".join(lines)


def build_planner_prompt(
    bb: Blackboard,
    user_message: str,
    candidates: list[str],
    last_agent: Optional[str] = None,
    last_reply_tail: Optional[str] = None,
    lang: Optional[str] = None,
) -> str:
    """Build the prompt for the planner LLM."""
    t = _texts(lang)
    hints = _route_hints(lang)
    cand_lines = "\n".join(f'  - "{c}": {hints.get(c, c)}' for c in candidates)
    goal_line = t["section_goal"].format(value=bb.goal) if bb.goal else ""
    prev_block = ""
    continuity_rule = ""
    if last_agent:
        tail = t["prev_tail"].format(value=last_reply_tail) if last_reply_tail else ""
        prev_block = t["section_prev"].format(agent=last_agent, tail=tail)
        continuity_rule = t["rule_continuity"]
    rule_n = 3 if last_agent else 2
    return (
        t["planner_role"]
        + f"{t['section_state']}{_state_summary(bb, lang)}{goal_line}\n"
        + prev_block
        + t["section_request"].format(value=user_message)
        + t["section_candidates"]
        + f"{cand_lines}\n"
        + t["section_rules"]
        + t["rule_explicit_intent"]
        + continuity_rule
        + t["rule_candidates_only"].format(n=rule_n)
        + t["rule_json_only"].format(n=rule_n + 1)
        + t["planner_json_example"]
    )


def parse_planner_decision(llm_text: str, candidates: list[str]) -> Optional[str]:
    """Extract the next node name from the planner response, or None.

    Order: (1) a ``{"next": ...}`` JSON object, (2) a word-boundary scan of the
    text. Anything but exactly one match gives None and the caller falls back.
    """
    if not llm_text or not candidates:
        return None
    cand_set = set(candidates)

    # (1) Prefer the JSON object, even wrapped in a code block or prose.
    for m in re.finditer(r'\{[^{}]*"next"[^{}]*\}', llm_text):
        try:
            val = json.loads(m.group(0)).get("next")
        except (ValueError, TypeError):
            continue
        if isinstance(val, str) and val in cand_set:
            return val

    # (2) Word-boundary match in the text, accepted only when unique.
    matched = [c for c in candidates if re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", llm_text)]
    if len(matched) == 1:
        return matched[0]
    return None
