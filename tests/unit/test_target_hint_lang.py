"""Tool hints must be English and must not translate the machine tokens.

The guidance the agent puts in ``_next_step`` is read by the LLM and copied
verbatim into the next tool call::

    Call detect_document_type(raw_dataset_folder='dataset_...').
    The target folder is '...' — do not pick a different one.

Two earlier gaps let Korean survive here:
  1. The scan in ``test_agent_runtime_texts`` only covers ``modules/agents``,
     and these strings live in ``server/core/corpus_target.py``.
  2. The call site is ``{"_next_step": target_hint(...)}``, so following the
     variable through the AST never reaches another module's return value.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from server.core.corpus_target import target_hint as corpus_hint
from server.core.kbd_target import target_hint as kbd_hint

REPO = Path(__file__).resolve().parents[2]
BOUNDARY_SPEC = REPO / "modules/agents/specialists/boundary_specialist.py"

_HANGUL = re.compile(r"[\uac00-\ud7a3]")

#: Machine contract: the LLM copies these straight into a tool call.
#: Translating one means the tool is never found.
_TOOL_TOKENS = {
    "corpus": "detect_document_type",
    "kbd": "get_gpu_info",
}


# ══════════════════════════════════════════════════════════════════
# 1. No Korean anywhere in the hints
# ══════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════
# 2. Machine contracts are never translated
# ══════════════════════════════════════════════════════════════════


class TestToolTokensPreserved:
    def test_corpus_tool_name(self):
        assert _TOOL_TOKENS["corpus"] in corpus_hint("d1", ["d1"])
        assert _TOOL_TOKENS["corpus"] in corpus_hint(None, ["d1"])

    def test_kbd_tool_name(self):
        assert _TOOL_TOKENS["kbd"] in kbd_hint("c1")
        assert _TOOL_TOKENS["kbd"] in kbd_hint(None)

    def test_argument_name(self):
        assert "raw_dataset_folder=" in corpus_hint("d1", ["d1"])

    def test_the_folder_name_is_carried_verbatim(self):
        """The LLM copies this value into a tool argument, so it must not be altered."""
        assert "dataset_20260916_061101" in corpus_hint("dataset_20260916_061101", [])


# ══════════════════════════════════════════════════════════════════
# 3. Wiring: the specialist passes the execution language through
# ══════════════════════════════════════════════════════════════════


def _passes_lang(path: Path, func: str) -> bool:
    """Whether a `func(...)` call passes `self._lang` as an argument."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func):
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if isinstance(arg, ast.Attribute) and arg.attr == "_lang":
                return True
        return False
    return False


class TestWiring:
    def test_boundary_specialist(self):
        assert _passes_lang(BOUNDARY_SPEC, "target_hint"), "target_hint is not passed self._lang"


# ══════════════════════════════════════════════════════════════════
# 4. No Korean literal comes back into these modules
# ══════════════════════════════════════════════════════════════════


class TestNoRegression:
    @pytest.mark.parametrize("rel", ["server/core/corpus_target.py", "server/core/kbd_target.py"])
    def test_no_korean_in_returned_strings(self, rel):
        """Docstrings and comments are prose; only the returned strings are checked."""
        src = (REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        doc_lines: set = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                if isinstance(n.value.value, str) and n.end_lineno:
                    doc_lines.update(range(n.lineno, n.end_lineno + 1))

        offenders = [
            i
            for i, line in enumerate(src.splitlines(), 1)
            if _HANGUL.search(line) and i not in doc_lines and not line.lstrip().startswith("#")
        ]
        assert not offenders, f"{rel}: Korean literals on lines {offenders}"
