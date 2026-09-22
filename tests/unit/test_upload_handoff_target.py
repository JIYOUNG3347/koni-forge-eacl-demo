"""The upload handoff must branch on the backend target, not the UI's name for it.

The front end calls its two upload sections "gen" and "train", and maps them to
the backend targets `raw` and `processed` before sending. The handoff builder
compared against `"gen"`, a value the API never receives, so every upload —
including a folder of PDFs — was handed the "a QA dataset was uploaded, go
train" instruction.

Checked at source level: the router needs fastapi, which a CI venv lacks.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROUTER = REPO / "server" / "routers" / "data.py"


def _handoff_body() -> str:
    src = DATA_ROUTER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_upload_handoff":
            return ast.unparse(node)
    raise AssertionError("_upload_handoff not found")


def test_it_branches_on_a_real_backend_target():
    body = _handoff_body()
    assert "'raw'" in body or '"raw"' in body
    assert "'gen'" not in body and '"gen"' not in body, "compares against the front-end name"


def test_source_documents_get_the_indexing_instruction():
    body = _handoff_body()
    # The raw branch must lead to the documents message, not the QA one.
    raw_index = body.index("raw")
    assert body.index("upload_docs_next") > raw_index
    assert body.index("upload_qa_next") > body.index("upload_docs_next")


def test_the_two_messages_say_different_things():
    from server.core.agent_handoff import handoff

    docs = handoff("upload_docs_next", "en", "folder_x")
    qa = handoff("upload_qa_next", "en", "folder_x")
    assert docs != qa
    assert "indexing" in docs.lower()
    assert "qa dataset" in qa.lower()
