"""Every stanza.Pipeline(...) instantiation in the codebase MUST pass
`download_method=DownloadMethod.NONE` and `model_dir=` sourced from
STANZA_RESOURCES_DIR. Otherwise the default DOWNLOAD_RESOURCES will
fetch resources_*.json over the network — which hangs forever in
KISTI's closed-network deployment.

These tests enforce the invariant for ALL sites at once, so any future
site automatically inherits the check.
"""

import re
from pathlib import Path

# Every site that calls stanza.Pipeline(...). Add new sites here as they
# appear — the parametrized tests below run against each.
PIPELINE_SITES = [
    Path("modules/agents/restructure_algo/dlmax_rewriter.py"),
    Path("modules/preprocessing/twist_algo.py"),
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _pipeline_call_blocks(src: str) -> list:
    """Yield every `stanza.Pipeline(...)` call expression (multi-line).

    Skips occurrences inside comments by requiring an assignment-style
    use (e.g. `nlp = stanza.Pipeline(` or `self.nlp = stanza.Pipeline(`).
    Returns the slice from the call's start through the matching close
    paren so kwargs can be asserted on.
    """
    blocks = []
    # Match `<identifier or self.x> = stanza.Pipeline(` to skip in-comment
    # references like "stanza.Pipeline() calls download_resources_json".
    for m in re.finditer(r"(?:\w+(?:\.\w+)*)\s*=\s*stanza\.Pipeline\(", src):
        start = m.start()
        # find call open paren
        open_idx = src.find("(", m.end() - 1)
        assert open_idx != -1
        depth = 0
        pos = open_idx
        end = pos
        while pos < len(src):
            ch = src[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
            pos += 1
        blocks.append(src[start : end + 1])
    return blocks


class TestEveryPipelineSiteIsOfflineSafe:
    """For each site, every Pipeline() call must pass offline kwargs."""

    def test_all_known_sites_are_listed(self):
        """Defend against new stanza.Pipeline() sites being added without a
        corresponding entry in PIPELINE_SITES — that would silently
        regress offline safety."""
        import subprocess

        result = subprocess.run(
            ["grep", "-rln", "stanza.Pipeline", "--include=*.py", "modules/", "celery_app/", "server/", "api.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Filter out tests/ and __pycache__
        found = {
            Path(line)
            for line in result.stdout.splitlines()
            if line and "__pycache__" not in line and not line.startswith("tests/")
        }
        listed = set(PIPELINE_SITES)
        missing = found - listed
        assert not missing, (
            f"new stanza.Pipeline site(s) found that aren't in "
            f"PIPELINE_SITES: {missing}. Add them to the list so this "
            f"suite enforces offline safety on them too."
        )
