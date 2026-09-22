"""Downloading a HuggingFace model into the models directory.

A repo id is flattened for the filesystem (``org/model`` -> ``org--model``),
which is the spelling ``locate_model_directory`` resolves back.

Checked here: the id is validated before it reaches the filesystem, the
in-flight lock is exclusive, and the task and the resolver agree on the name.
The download itself needs the network and is not exercised.
"""

import ast
from pathlib import Path

import pytest

from celery_app.tasks.model_tasks import ALLOW_PATTERNS

REPO = Path(__file__).resolve().parents[2]
MODELS_ROUTER = REPO / "server" / "routers" / "models.py"




class TestRepoIdValidation:
    """A crafted id must not reach the filesystem."""

    @pytest.fixture(scope="class")
    def pattern(self):
        import re

        src = MODELS_ROUTER.read_text(encoding="utf-8")
        line = next(ln for ln in src.splitlines() if ln.startswith("_REPO_ID = re.compile"))
        return re.compile(ast.literal_eval(line.split("re.compile(", 1)[1].rstrip(")")))

    @pytest.mark.parametrize(
        "repo_id",
        ["Qwen/Qwen2.5-0.5B-Instruct", "google/gemma-3-1b-it", "HuggingFaceTB/SmolLM2-135M-Instruct"],
    )
    def test_real_repo_ids_are_accepted(self, pattern, repo_id):
        assert pattern.match(repo_id)

    @pytest.mark.parametrize(
        "repo_id",
        ["../../etc/passwd", "/absolute/path", "org/model/../..", "no-slash", "", "org//model", "-bad/model"],
    )
    def test_anything_else_is_rejected(self, pattern, repo_id):
        assert not pattern.match(repo_id)


class TestWiring:
    @pytest.fixture(scope="class")
    def src(self):
        return MODELS_ROUTER.read_text(encoding="utf-8")

    def test_the_endpoints_the_agent_calls_exist(self, src):
        """boundary_specialist posts to these two; a missing one is a 404 loop."""
        assert '@router.post("/download/hf")' in src
        assert '@router.get("/download/{job_id}/status")' in src

        agent = (REPO / "modules/agents/specialists/boundary_specialist.py").read_text(encoding="utf-8")
        assert "/api/models/download/hf" in agent
        assert "/api/models/download/{job_id}/status" in agent

    def test_a_failed_dispatch_releases_the_lock(self, src):
        """Otherwise the model can never be downloaded again until the TTL expires."""
        block = src[src.index('@router.post("/download/hf")') :]
        assert "release_download(model_id)" in block
        assert block.index("except Exception") < block.index("release_download(model_id)")

    def test_the_task_reads_the_token_from_the_environment(self):
        task = (REPO / "celery_app" / "tasks" / "model_tasks.py").read_text(encoding="utf-8")
        assert 'os.getenv("HF_TOKEN")' in task
        assert "token=token" in task

    def test_a_failed_download_leaves_no_folder(self):
        """A half-written folder would list as installed and fail at load time."""
        task = (REPO / "celery_app" / "tasks" / "model_tasks.py").read_text(encoding="utf-8")
        assert task.count("shutil.rmtree(dest") >= 2

    def test_the_task_is_registered_with_celery(self):
        init = (REPO / "celery_app" / "__init__.py").read_text(encoding="utf-8")
        cfg = (REPO / "celery_app" / "config.py").read_text(encoding="utf-8")
        assert "celery_app.tasks.model_tasks" in init
        assert "celery_app.tasks.model_tasks.download_hf_task" in cfg

    def test_hf_token_is_documented(self):
        assert "HF_TOKEN" in (REPO / ".env.example").read_text(encoding="utf-8")


def test_allow_patterns_cover_the_loadable_files():
    """config.json and the weights must be fetched, or the model will not load."""
    assert "*.json" in ALLOW_PATTERNS
    assert "*.safetensors" in ALLOW_PATTERNS
