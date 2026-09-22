"""The centre's official English name is **Large-scale AI Research Center**.

Three different spellings used to coexist in the repository. A proper noun has
to be the agreed spelling, not merely a correct translation, and a mismatch
stays silent until somebody reads it — which matters most in LICENSE, a legal
document. So a test holds it.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

OFFICIAL = "Large-scale AI Research Center"

#: Files carrying the name. Leaving one out lets that file drift silently.
FILES = ["README.md", "LICENSE", "UI/src/components/LoginPage.tsx"]


@pytest.mark.parametrize("rel", FILES)
def test_official_name_appears(rel):
    body = (REPO / rel).read_text(encoding="utf-8")
    assert OFFICIAL in body, f"{rel} does not contain the official English name"


@pytest.mark.parametrize("rel", FILES)
def test_no_wrong_variant(rel):
    """No case variant or old name is left."""
    body = (REPO / rel).read_text(encoding="utf-8")
    wrong = []
    for m in re.finditer(r"[A-Za-z][A-Za-z\- ]*AI Research(?: Center)?", body):
        name = m.group(0).strip()
        if "AI Research" in name and name != OFFICIAL and "Hyperscale" not in name:
            # A partial match may just be a fragment of the official name.
            if OFFICIAL.endswith(name) or OFFICIAL.startswith(name):
                continue
            wrong.append(name)
        if "Hyperscale" in name:
            wrong.append(name)
    assert wrong == [], f"{rel}: spellings differing from the official name — {wrong}"


def test_hyperscale_is_gone_from_the_whole_repo():
    """The old name must be gone from everything that ships."""
    hits = []
    # Runtime and build data that is not tracked and does not ship; scanning it
    # would also make the suite several times slower.
    skip = {".git", ".venv", "node_modules", "storage", ".mypy_cache", "build", "dist", ".vite"}
    for path in REPO.rglob("*"):
        if not path.is_file() or any(p in skip for p in path.parts):
            continue
        # Tests name the old spelling on purpose, to explain what was fixed.
        # Only what ships is checked.
        rel_parts = path.relative_to(REPO).parts
        if "__tests__" in rel_parts or rel_parts[0] == "tests":
            continue
        if path.suffix.lower() in {".png", ".jpg", ".pdf", ".pptx", ".ico", ".woff", ".woff2"}:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "Hyperscale" in body:
            hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"the old name 'Hyperscale' is still present: {hits}"
