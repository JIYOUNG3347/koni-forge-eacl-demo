"""KBD probing — asking a base model the probe questions closed-book.

The run happens on the worker (loading a model would block the event loop and
compete with training), so this covers the task itself plus the two endpoints
the boundary agent drives it through.
"""

import ast
from pathlib import Path

import pytest

from celery_app.tasks import kbd_tasks
from celery_app.tasks.kbd_tasks import kbd_probe_task

REPO = Path(__file__).resolve().parents[2]
HF_ROUTER = REPO / "server" / "routers" / "hf.py"


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the model with an echo, and record the load / release calls."""
    calls: dict = {"activated": [], "released": 0}

    import pipelines.dialog as dialog

    monkeypatch.setattr(dialog, "activate_model", lambda name: calls["activated"].append(name))
    monkeypatch.setattr(dialog, "release_loaded_model", lambda: calls.update(released=calls["released"] + 1))
    monkeypatch.setattr(dialog, "generate_response", lambda q, **kw: f"answer to {q}")
    monkeypatch.setattr(kbd_tasks, "update_job", lambda *a, **kw: None)
    monkeypatch.setattr(kbd_tasks, "sys_log", lambda *a, **kw: None)

    import server.core.gpu_admission as admission

    monkeypatch.setattr(admission, "gate_or_retry", lambda *a, **kw: None)
    return calls


def run(**params):
    return kbd_probe_task.apply(kwargs={"job_id": "job-1", **params}, throw=True).get()


PROBES = [
    {"question": "What is KONI?", "ground_truth": "a Korean LLM", "category": "models"},
    {"question": "Who runs KISTI?", "ground_truth": "the government", "category": "org"},
]


class TestArguments:
    def test_a_model_name_is_required(self, stub_model):
        with pytest.raises(ValueError):
            run(model_name="  ", probes=PROBES)

    def test_probes_are_required(self, stub_model):
        with pytest.raises(ValueError):
            run(model_name="m", probes=[])


class TestProbing:
    def test_every_probe_is_answered(self, stub_model):
        result = run(model_name="m", probes=PROBES)
        assert result["total"] == 2
        assert result["errors"] == 0
        assert [r["model_answer"] for r in result["results"]] == [
            "answer to What is KONI?",
            "answer to Who runs KISTI?",
        ]

    def test_the_model_is_loaded_once_and_released(self, stub_model):
        run(model_name="m", probes=PROBES)
        assert stub_model["activated"] == ["m"]
        assert stub_model["released"] == 1

    def test_the_model_is_released_even_when_loading_fails(self, stub_model, monkeypatch):
        import pipelines.dialog as dialog

        monkeypatch.setattr(dialog, "activate_model", lambda name: (_ for _ in ()).throw(RuntimeError("no weights")))
        with pytest.raises(RuntimeError):
            run(model_name="m", probes=PROBES)
        assert stub_model["released"] == 1

    def test_one_failed_probe_does_not_lose_the_others(self, stub_model, monkeypatch):
        import pipelines.dialog as dialog

        def flaky(q, **kw):
            if "KISTI" in q:
                raise RuntimeError("out of memory")
            return "ok"

        monkeypatch.setattr(dialog, "generate_response", flaky)
        result = run(model_name="m", probes=PROBES)
        assert result["errors"] == 1
        assert result["total"] == 2
        assert result["results"][0]["model_answer"] == "ok"
        assert result["results"][1]["model_answer"] == ""
        assert "out of memory" in result["results"][1]["error"]

    def test_an_empty_question_is_an_error_not_a_crash(self, stub_model):
        result = run(model_name="m", probes=[{"question": "", "ground_truth": "x", "category": "c"}])
        assert result["errors"] == 1
        assert result["results"][0]["model_answer"] == ""


class TestJudgeContract:
    """The judge and the boundary map read these fields off each result."""

    def test_each_result_carries_what_the_judge_scores(self, stub_model):
        from server.core.judge_prompts import kbd_judge_prompt

        entry = run(model_name="m", probes=PROBES)["results"][0]
        assert {"question", "ground_truth", "category", "model_answer"} <= set(entry)
        prompt = kbd_judge_prompt([entry])
        assert "What is KONI?" in prompt and "answer to What is KONI?" in prompt


class TestWiring:
    def test_the_task_is_registered_with_the_worker(self):
        import celery_app

        assert "celery_app.tasks.kbd_tasks" in celery_app.TASK_MODULES

    def test_it_runs_on_the_gpu_queue(self):
        from celery_app.config import CeleryConfig

        assert CeleryConfig.task_routes[kbd_probe_task.name]["queue"] == "gpu"

    def test_the_lease_ledger_knows_a_probe_occupies_the_device(self):
        from server.core.gpu_resources import classify, infer_consumer_kind

        kind = infer_consumer_kind(kbd_probe_task.name)
        assert kind == "kbd_probe"
        assert classify(kind) is not None


@pytest.fixture(scope="module")
def routes():
    """(method, path) of every route declared in the router."""
    tree = ast.parse(HF_ROUTER.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.args:
                found.append((dec.func.attr, ast.literal_eval(dec.args[0])))
    return found


class TestEndpoints:
    """The boundary agent posts to /api/hf/probe and polls /api/hf/probe/{job_id}."""

    def test_both_probe_routes_exist(self, routes):
        assert ("post", "/probe") in routes
        assert ("get", "/probe/{job_id}") in routes

    def test_the_status_route_returns_the_fields_the_agent_polls(self):
        src = HF_ROUTER.read_text(encoding="utf-8")
        body = src[src.index('@router.get("/probe/{job_id}")'):]
        for field in ("status", "progress", "message", "error", "result"):
            assert f'"{field}"' in body


class TestSavedRecord:
    """A verdict can only be rechecked if the answer it was given for is kept."""

    def test_the_saved_record_keeps_the_answer_and_the_reasoning(self):
        """generate_boundary_map projects scored results into ``per_question``."""
        src = (REPO / "modules" / "agents" / "specialists" / "boundary_specialist.py").read_text(encoding="utf-8")
        block = src[src.index("per_question = ["):]
        block = block[: block.index("]")]
        for field in ("question", "ground_truth", "category", "model_answer", "verdict", "confidence", "explanation"):
            assert f'"{field}"' in block, f"{field} dropped from the saved KBD record"


class TestCategoryGrouping:
    """A category holding one probe cannot be judged, so the extractor is told
    how many to use rather than left to invent one per question."""

    SRC = (REPO / "modules" / "agents" / "specialists" / "boundary_specialist.py").read_text(encoding="utf-8")

    @staticmethod
    def n_categories(count: int) -> int:
        """The formula in _extract_key_concepts."""
        from server.core.kbd_verdict import MIN_CATEGORY_PROBES

        return max(1, min(6, count // max(2 * MIN_CATEGORY_PROBES, 1)))

    @pytest.mark.parametrize("count", [5, 6, 10, 13, 20, 30, 50])
    def test_every_category_clears_the_judging_floor(self, count):
        from server.core.kbd_verdict import MIN_CATEGORY_PROBES

        assert count / self.n_categories(count) >= MIN_CATEGORY_PROBES

    def test_the_prompt_fixes_the_category_count(self):
        assert "Decide on exactly {n_categories} sub-categories" in self.SRC
        assert "Do not invent a category per question" in self.SRC

    def test_an_unjudgeable_breakdown_is_stated_in_the_result(self):
        """Otherwise the agent is told to report weak categories from an empty list."""
        assert '"category_note"' in self.SRC
        assert "Do not report weak categories for this run." in self.SRC

    def test_the_agent_is_told_not_to_name_a_category_it_was_not_given(self):
        assert "Never name a category that is not in that output." in self.SRC
