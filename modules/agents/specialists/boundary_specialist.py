"""
KONI-Forge BoundarySpecialist — Knowledge Boundary Detection agent

Roles:
- Use RAG-indexed documents as ground truth
- Auto-extract 50 key concepts/facts from documents
- Execute probe queries on base model
- Auto-score response quality (document = ground truth)
- Generate knowledge boundary map (0~100% spectrum)
- Auto-recommend RAG / Fine-Tuning / Hybrid path
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

from server.core.agent_tool_errors import error_text as _err
from server.core.dataset_paths import chroma_dir, iter_dataset_dirs
from server.core.judge_prompts import kbd_judge_prompt
from server.core.kbd_verdict import MIN_CATEGORY_PROBES as _MIN_CATEGORY_PROBES
from server.core.kbd_verdict import STATUS_BOUNDARY as _STATUS_BOUNDARY
from server.core.kbd_verdict import STATUS_INSUFFICIENT as _STATUS_INSUFFICIENT
from server.core.kbd_verdict import STATUS_UNKNOWN as _STATUS_UNKNOWN
from server.core.kbd_verdict import base_path as _verdict_base_path
from server.core.kbd_verdict import category_status as _category_status
from server.core.kbd_verdict import resolve_path as _resolve_verdict_path

from ..foundation import AgentBase, AgentEvent, AgentPhase, ChainSignal

logger = logging.getLogger("agent.boundary")
INTERNAL_API_URL = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
_INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")


def _internal_headers() -> dict:
    return {"X-Internal-Token": _INTERNAL_TOKEN}


class BoundarySpecialist(AgentBase):
    name = "boundary"
    description = "Knowledge Boundary Detection — measures what the model already knows about the corpus and recommends RAG or fine-tuning"
    default_autonomy = "guided"
    system_prompt = """You are the Knowledge Boundary Detection (KBD) specialist of KONI-Forge.

Role: use the documents indexed in RAG as ground truth, measure how much of that knowledge the base model already has, and recommend one of three paths: RAG only, Fine-Tuning, or Hybrid.

== KBD procedure (call the tools in this order) ==

Step 1: get_indexed_documents
  - List the documents indexed in RAG.
  - If nothing is indexed, tell the user to upload and index documents first and stop.

Step 2: get_gpu_info
  - Check the available GPU VRAM.

Step 3: list_local_models, then choose a model
  - List the locally installed models and pick one that fits the VRAM:
    * up to 8 GB  -> 0.5B–1B models
    * 8–16 GB     -> 1B–3B models
    * 16 GB+      -> 3B–7B models
  - If the system prompt ends with "[Agent mode: auto]": do not ask. Pick the largest local model that fits and continue with Step 4 after one line such as "Running KBD with gemma-3-1b-it."
  - Otherwise (recommend-then-confirm mode): present the options in this form and ask which model to use:

    **GPU**: RTX 3090 (24 GB VRAM)

    **Option 1: Qwen--Qwen2-0.5B-Instruct (installed)**
    Ready to use right away.

    **Option 2: gemma-3-1b-it (installed)**
    A 1B model gives a more accurate measurement.

  - The chosen model becomes the fine-tuning target later, so the size matters.
  - If the model must be downloaded first, call download_model, then continue.

Step 4: extract_key_concepts (doc_id=<document id>, doc_name=<document name>, count=20)
  - The tool fetches the chunks itself and extracts 20 key concepts as question + ground_truth + category.

Step 5: run_knowledge_probes
  - probes: pass the "probes" array from Step 4 unchanged.
  - model_name: the model chosen in Step 3.

Step 6: score_probe_responses
  - probe_results: pass only the "results" array from Step 5 (drop model, total and other fields).

Step 7: generate_boundary_map
  - scored_results: the "scored_results" array from Step 6.
  - model_name and doc_name: what was used above.
  - Produces the final knowledge-boundary map and the RAG / Fine-Tuning / Hybrid recommendation.

== Rules ==
- Run Steps 1 → 7 in order. Never skip a step; feed each step's output into the next.
- When asked to "run KBD", start with Step 1 immediately, without a preamble.
- Once the model is decided (by the user or automatically), run Steps 4 → 7 back to back without stopping.
- Do not summarise Step 6 yourself — always call generate_boundary_map; its output is the final analysis.
- If a step fails, stop there and report the error to the user.

== Recommendation thresholds ==
- knowledge coverage ≥ 70 %  -> RAG is sufficient
- knowledge coverage ≤ 50 %  -> Fine-Tuning is needed
- 50–70 %                    -> Hybrid (RAG + LoRA)

== After the verdict ==
Report the result only; the system decides the next step.
- RAG verdict: "The model already covers this material — retrieval alone is enough."
- Fine-Tuning verdict: coverage < 30 % → "Full fine-tuning is recommended."; 30–50 % → "LoRA fine-tuning is suitable."; hallucination rate ≥ 20 % → "The hallucination rate is high — full fine-tuning is strongly recommended." Report coverage, hallucination rate and weak categories in detail.
- Weak categories come from generate_boundary_map only. When its `category_note` says none could be judged, or the weak category list is empty, say that the per-category breakdown is unavailable. Never name a category that is not in that output.
- Hybrid verdict: "RAG plus lightweight fine-tuning (LoRA) is recommended."
Do not ask "Shall I start training?" — the orchestrator proposes the next step."""

    async def run(self, user_message: str):
        """Run the full KBD pipeline, steps 1 to 7, in code.
        Keeps the 7-step tool chain stable across models.
        Always uses the hardcoded pipeline (_run_kbd_pipeline).
        - Auto: every step runs automatically (the model is picked by free memory)
        - Guided: stops at step 3 to confirm the model, then runs steps 4 to 7
        """
        _ = self.client  # lazy init, which also sets _network_mode
        # Reset the tool-call flag for a new analysis.
        if not self.conversation:
            self._list_local_models_called = False

        # effective_mode ignores the dispatcher's runtime override (set_autonomy("auto"))
        # and follows the mode the user actually configured, so a guided user still
        # gets the model-selection prompt during a chain.
        _cfg = AgentBase._load_system_config()
        effective_mode = self._load_user_autonomy() or _cfg.get("agent_autonomy") or self.default_autonomy or "guided"

        # Guided re-entry: the user confirmed the model, so resume at step 4.
        if hasattr(self, "_kbd_guided_state") and self._kbd_guided_state:
            import re as _re

            _confirm_words = {"yes", "y", "ok", "okay", "go", "proceed", "start", "sure"}
            _msg_lower = user_message.strip().lower()
            state = self._kbd_guided_state
            self._kbd_guided_state = None

            candidates = state.get("candidates", [])
            candidate_names = [c[0] for c in candidates]
            download_opt = state.get("download_opt")  # (model_name, size) or None

            # Model chosen by "Option N".
            _opt_match = _re.match(r"option\s*(\d+)", _msg_lower)
            if _opt_match:
                opt_idx = int(_opt_match.group(1)) - 1  # 0-based
                if download_opt and opt_idx == len(candidates):
                    # Download option chosen.
                    dl_name, _ = download_opt
                    self.conversation.append({"role": "user", "content": user_message})
                    self.status = AgentPhase.THINKING
                    yield AgentEvent(
                        role="system",
                        content=self._rt("model_downloading", model=dl_name),
                        status=AgentPhase.THINKING,
                        agent_name=self.name,
                    )
                    try:
                        dl_result = await self._download_model({"model_id": dl_name})
                        dl_data = json.loads(dl_result)
                        if dl_data.get("error"):
                            yield AgentEvent(
                                role="assistant",
                                content=self._rt("download_failed", error=dl_data["error"]),
                                status=AgentPhase.ERROR,
                                agent_name=self.name,
                            )
                            self.status = AgentPhase.IDLE
                            return
                        state["selected_model"] = dl_name
                        yield AgentEvent(
                            role="system",
                            content=self._rt("download_done", model=dl_name),
                            status=AgentPhase.THINKING,
                            agent_name=self.name,
                        )
                    except Exception as e:
                        yield AgentEvent(
                            role="assistant",
                            content=self._rt("download_error", error=str(e)),
                            status=AgentPhase.ERROR,
                            agent_name=self.name,
                        )
                        self.status = AgentPhase.IDLE
                        return
                elif 0 <= opt_idx < len(candidates):
                    state["selected_model"] = candidates[opt_idx][0]
                # Out of range — keep the recommended model.

            # Check whether the user typed a model name directly.
            for name in candidate_names:
                if name.lower() in _msg_lower:
                    state["selected_model"] = name
                    break

            _already_appended = download_opt and _opt_match and int(_opt_match.group(1)) - 1 == len(candidates)
            if (
                any(w in _msg_lower for w in _confirm_words)
                or any(n.lower() in _msg_lower for n in candidate_names)
                or _opt_match
            ):
                if not _already_appended:
                    self.conversation.append({"role": "user", "content": user_message})
                self.status = AgentPhase.THINKING
                try:
                    async for ev in self._run_kbd_pipeline(
                        resume_state=state,
                    ):
                        yield ev
                except Exception as e:
                    logger.error(f"KBD pipeline error: {e}", exc_info=True)
                    yield AgentEvent(
                        role="assistant",
                        content=self._rt("kbd_pipeline_error", error=str(e)),
                        status=AgentPhase.ERROR,
                        agent_name=self.name,
                    )
                finally:
                    self.status = AgentPhase.IDLE
                return

        # A conversation already in progress (tool results present) uses the normal run.
        has_tool_results = any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
            for m in self.conversation
        )
        if has_tool_results:
            async for event in super().run(user_message):
                yield event
            return

        self.conversation.append({"role": "user", "content": user_message})
        self.status = AgentPhase.THINKING

        try:
            async for ev in self._run_kbd_pipeline(
                guided=(effective_mode == "guided"),
            ):
                yield ev
        except Exception as e:
            logger.error(f"KBD pipeline error: {e}", exc_info=True)
            yield AgentEvent(
                role="assistant",
                content=self._rt("kbd_pipeline_error", error=str(e)),
                status=AgentPhase.ERROR,
                agent_name=self.name,
            )
        finally:
            self._pipeline_mode = False
            self.status = AgentPhase.IDLE

    async def _run_kbd_pipeline(self, guided: bool = False, resume_state: dict = None):
        """Run the seven-step KBD pipeline.

        guided=False: every step runs automatically.
        guided=True: stops at step 3 for model confirmation and resumes from resume_state.
        resume_state: the step 1 to 3 results (doc_id, doc_name, selected_model).
        """
        import uuid as _uuid

        self._pipeline_mode = True  # marks the hardcoded pipeline as running

        async def _run_step(tool_name: str, tool_input: dict, step_label: str):
            """Run a tool, emit the event and record it in the conversation.
            Yields a thinking event every second so SSE stays alive during a long tool call.
            """
            yield AgentEvent(
                role="assistant",
                content=self._rt("tool_call", tool=tool_name),
                status=AgentPhase.TOOL_USE,
                agent_name=self.name,
                tool_name=tool_name,
                tool_input=tool_input,
            )
            # Run the tool as a background task, yielding keepalives until it finishes.
            # A THINKING event with empty content keeps the UI spinner without rendering a message.
            _tool_task = asyncio.ensure_future(self.execute_tool(tool_name, tool_input))
            while not _tool_task.done():
                await asyncio.sleep(1)
                if not _tool_task.done():
                    yield AgentEvent(
                        role="assistant",
                        content="",
                        status=AgentPhase.TOOL_USE,
                        agent_name=self.name,
                        tool_name=tool_name,
                    )
            try:
                result = _tool_task.result()
            except Exception as e:
                result = json.dumps({"error": str(e)})

            # Parse the chain signal.
            parsed_chain = None
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "__chain_signal__" in parsed:
                    sig = parsed["__chain_signal__"]
                    parsed_chain = ChainSignal(stage_completed=sig["stage_completed"], metadata=sig.get("metadata", {}))
            except Exception:
                pass

            yield AgentEvent(
                role="tool_result",
                content=result[:500],
                status=AgentPhase.TOOL_USE,
                agent_name=self.name,
                tool_name=tool_name,
                tool_result=result[:500],
                chain_signal=parsed_chain,
            )
            # Record it in the conversation.
            fake_id = f"kbd_{_uuid.uuid4().hex[:8]}"
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": fake_id, "name": tool_name, "input": tool_input}],
                }
            )
            self.conversation.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": fake_id, "content": result}]}
            )
            _run_step._last_result = result
            _run_step._last_chain = parsed_chain

        _run_step._last_result = ""
        _run_step._last_chain = None

            # ── Guided re-entry: resume_state means steps 1 to 3 are done ──
        if resume_state:
            doc_id = resume_state["doc_id"]
            doc_name = resume_state["doc_name"]
            selected_model = resume_state["selected_model"]
            candidates = resume_state.get("candidates", [])
            yield AgentEvent(
                role="system",
                content=self._rt("kbd_resume_with_model", model=selected_model),
                status=AgentPhase.THINKING,
                agent_name=self.name,
            )
        else:
            # ── Step 1: get_indexed_documents ──
            yield AgentEvent(
                role="system", content=self._rt("kbd_start"), status=AgentPhase.THINKING, agent_name=self.name
            )
            async for ev in _run_step("get_indexed_documents", {}, "Step 1"):
                yield ev
            try:
                docs_data = json.loads(_run_step._last_result)
            except Exception:
                docs_data = {}
            if not docs_data.get("documents"):
                yield AgentEvent(
                    role="assistant",
                    content=self._rt("kbd_no_indexed_docs"),
                    status=AgentPhase.RESPONDING,
                    agent_name=self.name,
                )
                self.status = AgentPhase.IDLE
                return
            doc = docs_data["documents"][0]
            doc_id = doc.get("id", "")
            doc_name = doc.get("name", "")

            # ── Step 2: get_gpu_info ──
            async for ev in _run_step("get_gpu_info", {}, "Step 2"):
                yield ev

            # ── Step 3: list_local_models ──
            async for ev in _run_step("list_local_models", {}, "Step 3"):
                yield ev
            try:
                models_data = json.loads(_run_step._last_result)
            except Exception:
                models_data = {}
            models = models_data.get("models", [])
            if not models:
                yield AgentEvent(
                    role="assistant",
                    content=self._rt("kbd_no_base_model"),
                    status=AgentPhase.RESPONDING,
                    agent_name=self.name,
                )
                self.status = AgentPhase.IDLE
                return

            # Pick the best model that fits in the free device memory.
            import re as _re

            def _estimate_size(name: str) -> float:
                m = _re.search(r"(\d+\.?\d*)[Bb]", name)
                return float(m.group(1)) if m else 0.5

            vram_mb = 0
            for msg in reversed(self.conversation[-6:]):
                c = msg.get("content", "")
                if isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            try:
                                gd = json.loads(block["content"])
                                if "gpus" in gd:
                                    gpus = gd["gpus"]
                                    if isinstance(gpus, list) and gpus:
                                        vram_mb = gpus[0].get("memory_total", 0)
                                    break
                            except Exception:
                                pass
            max_size = 1.0 if vram_mb < 8000 else 3.0 if vram_mb < 16000 else 7.0
            candidates = [(m["name"], _estimate_size(m["name"])) for m in models]
            candidates = [(n, s) for n, s in candidates if s <= max_size]
            if not candidates:
                candidates = [(models[0]["name"], 0)]
            selected_model = max(candidates, key=lambda x: x[1])[0]

            # ── Guided: stop at step 3 to confirm the model ──
            if guided:
                # Free-memory text.
                vram_gb = round(vram_mb / 1024, 1) if vram_mb else 0
                gpu_text = f"**Device**: {vram_gb}GB free" if vram_gb else ""

                # Locally installed models, rendered as **Option N: Label**.
                option_lines = []
                for idx, (n, s) in enumerate(candidates, 1):
                    rec_tag = " <- recommended" if n == selected_model else ""
                    option_lines.append(f"**Option {idx}: {n}**\n{s}B model, installed locally{rec_tag}")

                # With memory to spare, offer to download a larger model.
                # KBD uses HuggingFace models, so the full repo id (org/model) is
                # required — a bare name such as "Qwen2.5-3B-Instruct" fails,
                # because the HF download needs the org prefix.
                _download_suggestions = {
                    8000: ("google/gemma-3-1b-it", 1.0),
                    16000: ("Qwen/Qwen2.5-3B-Instruct", 3.0),
                    24000: ("Qwen/Qwen2.5-7B-Instruct", 7.0),
                }
                download_opt = None
                for vram_threshold, (dl_name, dl_size) in sorted(_download_suggestions.items()):
                    if vram_mb >= vram_threshold and dl_size > max(s for _, s in candidates):
                        # Already local — do not offer it (compare on the repo id basename).
                        _dl_base = dl_name.split("/")[-1].lower()
                        if not any(_dl_base in n.lower() for n, _ in candidates):
                            download_opt = (dl_name, dl_size)
                if download_opt:
                    dl_name, dl_size = download_opt
                    next_idx = len(candidates) + 1
                    option_lines.append(
                        f"**Option {next_idx}: {dl_name}**\n{dl_size}B model, needs downloading (more accurate measurement)"
                    )

                ask_text = (
                    self._rt("kbd_model_pick_title")
                    + (f"{gpu_text}\n\n" if gpu_text else "")
                    + "\n\n".join(option_lines)
                    + self._rt("kbd_model_pick_ask")
                )
                logger.info(
                    f"[KBD-PIPELINE] Guided: yielding model selection message (len={len(ask_text)}, candidates={len(candidates)})"
                )
                self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": ask_text}]})
                self.status = AgentPhase.RESPONDING
                yield AgentEvent(role="assistant", content=ask_text, status=AgentPhase.RESPONDING, agent_name=self.name)
                # Save the pipeline state; run() resumes here after confirmation.
                self._kbd_guided_state = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "selected_model": selected_model,
                    "candidates": candidates,
                    "download_opt": download_opt,
                }
                self.status = AgentPhase.IDLE
                return

            yield AgentEvent(
                role="system",
                content=self._rt("kbd_run_with_model", model=selected_model),
                status=AgentPhase.THINKING,
                agent_name=self.name,
            )

        # ── Step 4: extract_key_concepts ──
        async for ev in _run_step(
            "extract_key_concepts", {"doc_id": doc_id, "doc_name": doc_name, "count": 20}, "Step 4"
        ):
            yield ev
        try:
            concepts_data = json.loads(_run_step._last_result)
        except Exception:
            concepts_data = {}
        probes = concepts_data.get("probes", [])
        if not probes:
            yield AgentEvent(
                role="assistant",
                content=self._rt("kbd_concepts_failed", error=concepts_data.get("error") or self._rt("unknown_error")),
                status=AgentPhase.RESPONDING,
                agent_name=self.name,
            )
            self.status = AgentPhase.IDLE
            return

        # ── Step 5: run_knowledge_probes (slow — show progress) ──
        yield AgentEvent(
            role="system",
            content=self._rt("kbd_probing", model=selected_model, count=len(probes)),
            status=AgentPhase.THINKING,
            agent_name=self.name,
        )
        async for ev in _run_step("run_knowledge_probes", {"probes": probes, "model_name": selected_model}, "Step 5"):
            yield ev
        try:
            probe_data = json.loads(_run_step._last_result)
        except Exception:
            probe_data = {}
        probe_results = probe_data.get("results", [])
        if not probe_results:
            yield AgentEvent(
                role="assistant",
                content=self._rt("kbd_probe_failed", error=probe_data.get("error") or self._rt("unknown_error")),
                status=AgentPhase.RESPONDING,
                agent_name=self.name,
            )
            self.status = AgentPhase.IDLE
            return

        # ── Step 6: score_probe_responses (LLM grading — slow) ──
        yield AgentEvent(
            role="system",
            content=self._rt("kbd_scoring", count=len(probe_results)),
            status=AgentPhase.THINKING,
            agent_name=self.name,
        )
        async for ev in _run_step("score_probe_responses", {"probe_results": probe_results}, "Step 6"):
            yield ev

        # ── Step 7: generate_boundary_map ──
        async for ev in _run_step(
            "generate_boundary_map", {"model_name": selected_model, "doc_name": doc_name}, "Step 7"
        ):
            yield ev

        # Final summary.
        try:
            result_data = json.loads(_run_step._last_result)
            rec = result_data.get("recommendation", {})
            kb = result_data.get("knowledge_boundary", {})
            summary = self._rt(
                "kbd_summary",
                model=selected_model,
                doc=doc_name,
                coverage=kb.get("knowledge_coverage_pct", "N/A"),
                hallucination=rec.get("hallucination_rate_pct", "N/A"),
                path=rec.get("path", "N/A"),
                detail=rec.get("detail", ""),
            )
        except Exception:
            summary = self._rt("kbd_done")

        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": summary}]})
        self.status = AgentPhase.RESPONDING
        yield AgentEvent(role="assistant", content=summary, status=AgentPhase.RESPONDING, agent_name=self.name)

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_indexed_documents",
                "description": "List the documents indexed in the RAG vector store along with their chunk information.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_gpu_info",
                "description": "Get the GPU status (total VRAM, used, free). This is the basis for choosing a model size.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "download_model",
                "description": "Download a HuggingFace model (e.g. Qwen/Qwen2.5-7B-Instruct). Waits until the download completes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "HuggingFace repo id (org/model-name form). e.g. Qwen/Qwen2.5-3B-Instruct, google/gemma-3-1b-it",
                        },
                    },
                    "required": ["model_id"],
                },
            },
            {
                "name": "retrieve_document_chunks",
                "description": "Search the chunk contents of a specific document. Used for extracting key concepts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID",
                        },
                        "sample_count": {
                            "type": "integer",
                            "description": "Number of chunks to fetch (default: 20)",
                        },
                    },
                    "required": ["doc_id"],
                },
            },
            {
                "name": "extract_key_concepts",
                "description": "Automatically fetch the document chunks by doc_id and extract key concepts and facts. Generates probe questions plus ground truth.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "ID of an indexed document (see get_indexed_documents)",
                        },
                        "doc_name": {
                            "type": "string",
                            "description": "Document name",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of key concepts to extract (default: 20)",
                        },
                    },
                    "required": ["doc_id", "doc_name"],
                },
            },
            {
                "name": "run_knowledge_probes",
                "description": "Run the extracted key concepts as probe queries against a local base model (storage/models). Loads the model automatically, tests whether it knows each concept, then unloads it.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "probes": {
                            "type": "array",
                            "description": "Probe list [{question, ground_truth, category}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "ground_truth": {"type": "string"},
                                    "category": {"type": "string"},
                                },
                            },
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Name of the local model to test (e.g. Qwen--Qwen2-0.5B-Instruct, gemma-3-1b-it)",
                        },
                    },
                    "required": ["probes", "model_name"],
                },
            },
            {
                "name": "score_probe_responses",
                "description": "Automatically grade the probe answers against the document (ground truth). Evaluates each answer's correctness, confidence and whether it hallucinated.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "probe_results": {
                            "type": "array",
                            "description": "Probe results [{question, ground_truth, model_answer, category}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "ground_truth": {"type": "string"},
                                    "model_answer": {"type": "string"},
                                    "category": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["probe_results"],
                },
            },
            {
                "name": "generate_boundary_map",
                "description": "Build a knowledge-boundary map from the grading results and recommend a RAG / FT / Hybrid path.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scored_results": {
                            "type": "array",
                            "description": "List of graded probe results",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "category": {"type": "string"},
                                    "verdict": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                            },
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model being analysed",
                        },
                        "doc_name": {
                            "type": "string",
                            "description": "Name of the document used for the analysis",
                        },
                    },
                    "required": ["scored_results", "model_name", "doc_name"],
                },
            },
            {
                "name": "list_local_models",
                "description": "List the local models in storage/models. Use this to choose the model to test with KBD.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_kbd_history",
                "description": "List previous KBD analysis results.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        if tool_name == "get_indexed_documents":
            return self._get_indexed_documents()
        elif tool_name == "get_gpu_info":
            return await self._get_gpu_info()
        elif tool_name == "download_model":
            return await self._download_model(tool_input)
        elif tool_name == "retrieve_document_chunks":
            return await self._retrieve_document_chunks(tool_input)
        elif tool_name == "extract_key_concepts":
            return await self._extract_key_concepts(tool_input)
        elif tool_name == "run_knowledge_probes":
            return await self._run_knowledge_probes(tool_input)
        elif tool_name == "score_probe_responses":
            return await self._score_probe_responses(tool_input)
        elif tool_name == "generate_boundary_map":
            return self._generate_boundary_map(tool_input)
        elif tool_name == "list_local_models":
            # prevent repeated calls in free mode
            if not getattr(self, "_pipeline_mode", False):
                if hasattr(self, "_list_local_models_called") and self._list_local_models_called:
                    return "The model list has already been retrieved. Do not call another tool; ask the user to choose a model based on the results above."
                self._list_local_models_called = True
            return await self._list_local_models()
        elif tool_name == "get_kbd_history":
            return self._get_kbd_history()
        return f"Unknown tool: {tool_name}"

    async def _get_gpu_info(self) -> str:
        """Device memory, used to recommend a model size."""
        from server.core import accelerator

        try:
            info = accelerator.probe()
            return json.dumps(
                {
                    "backend": info["backend"],
                    "gpus": info["gpus"],
                    "_next_step": self._rt("next_list_local_models"),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": _err("gpu_query_failed", self._lang, error=str(e)),
                    "_next_step": self._rt("next_list_local_models"),
                }
            )

    async def _download_model(self, params: Dict[str, Any]) -> str:
        """Download a HuggingFace model, then poll until it completes."""
        model_id = params["model_id"]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                res = await client.post(
                    f"{INTERNAL_API_URL}/api/models/download/hf",
                    json={"model_id": model_id},
                    headers=_internal_headers(),
                )

            if res.status_code == 409:
                return json.dumps({"error": _err("download_already_running", self._lang)})
            if res.status_code != 200:
                return json.dumps(
                    {
                        "error": _err(
                            "download_start_http_failed",
                            self._lang,
                            status=res.status_code,
                            detail=res.text[:200],
                        )
                    }
                )

            job_id = (res.json() or {}).get("job_id")
            if not job_id:
                return json.dumps({"error": _err("download_no_job_id", self._lang)})

            # Poll the job status.
            for _ in range(600):  # up to ~30 minutes at 3s intervals
                await asyncio.sleep(3)
                async with httpx.AsyncClient(timeout=10) as client:
                    status_res = await client.get(
                        f"{INTERNAL_API_URL}/api/models/download/{job_id}/status",
                        headers=_internal_headers(),
                    )
                if status_res.status_code != 200:
                    continue
                st = status_res.json()
                status = (st.get("status") or "").upper()
                if status == "SUCCESS":
                    return json.dumps(
                        {
                            "success": True,
                            "model_id": model_id,
                            "message": self._rt("res_model_download_done", model=model_id),
                        },
                        ensure_ascii=False,
                    )
                if status in ("FAILURE", "REVOKED"):
                    detail = st.get("error") or st.get("message") or _err("unknown_error", self._lang)
                    return json.dumps(
                        {
                            "error": _err("download_failed", self._lang, detail=detail),
                        },
                        ensure_ascii=False,
                    )

            return json.dumps({"error": _err("download_timeout", self._lang, minutes=30)})
        except Exception as e:
            return json.dumps({"error": _err("download_failed", self._lang, detail=str(e))})

    def _get_indexed_documents(self) -> str:
        """List indexed documents by walking live ChromaDB collections.

        Each upload folder produces its own collection (see celery index task);
        `doc_id` downstream is the collection name (= folder name).
        """
        chroma_path = str(chroma_dir())
        if not Path(chroma_path).exists():
            return json.dumps(
                {
                    "documents": [],
                    "message": self._rt("res_no_vectorstore_upload"),
                },
                ensure_ascii=False,
            )

        try:
            import chromadb

            client = chromadb.PersistentClient(path=chroma_path)
            docs = []
            for col in client.list_collections():
                try:
                    count = col.count()
                except Exception:
                    count = 0
                if count > 0:
                    docs.append({"id": col.name, "name": col.name, "chunks": count})
            if not docs:
                return json.dumps(
                    {
                        "documents": [],
                        "message": self._rt("res_no_indexed_docs_upload"),
                    },
                    ensure_ascii=False,
                )
            from server.core.kbd_target import select_kbd_target, target_hint

            docs, requested = select_kbd_target(docs, self._recent_user_text())
            if requested:
                logger.info(f"[BoundarySpecialist] KBD target specified: {requested}")
            payload = {
                "documents": docs,
                "count": len(docs),
                "_next_step": target_hint(requested, self._lang),
            }
            if requested:
                payload["requested_document"] = requested
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {
                    "documents": [],
                    "error": _err("index_list_failed", self._lang, error=str(e)),
                },
                ensure_ascii=False,
            )

    def _recent_user_text(self, limit: int = 3) -> str:
        """Recent user messages joined, used to detect a requested target dataset.

        The conversation is owned by AgentBase. On failure the empty string falls
        back to the previous behaviour (largest collection first).
        """
        try:
            texts = []
            for turn in reversed(self.conversation):
                if not isinstance(turn, dict) or turn.get("role") != "user":
                    continue
                content = turn.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                if isinstance(content, str) and content.strip():
                    texts.append(content)
                if len(texts) >= limit:
                    break
            return " ".join(texts)
        except Exception:  # noqa: BLE001 — a detection failure must not block KBD
            return ""

    async def _retrieve_document_chunks(self, params: Dict[str, Any]) -> str:
        """Return up to N chunks from the ChromaDB collection named `doc_id`.

        `doc_id` is the collection name (= raw folder name = upload dataset id).
        """
        doc_id = params["doc_id"]
        sample_count = int(params.get("sample_count", 20))

        chroma_path = str(chroma_dir())
        try:
            import chromadb

            # Retrieval specialist also opens the same path with default settings;
            # passing a divergent Settings triggers Chroma's "different settings"
            # singleton guard. Always use defaults here to stay compatible.
            client = chromadb.PersistentClient(path=chroma_path)
            try:
                collection = client.get_collection(name=doc_id)
            except Exception:
                return json.dumps(
                    {
                        "error": _err("collection_not_found", self._lang, doc_id=doc_id),
                    }
                )

            results = collection.get(limit=sample_count, include=["documents", "metadatas"])
            raw_docs = results.get("documents") or []
            raw_meta = results.get("metadatas") or []

            chunks = []
            for i, doc in enumerate(raw_docs):
                chunks.append(
                    {
                        "index": i,
                        "content": doc,
                        "metadata": raw_meta[i] if i < len(raw_meta) and raw_meta[i] else {},
                    }
                )

            return json.dumps(
                {
                    "doc_id": doc_id,
                    "chunks": chunks,
                    "count": len(chunks),
                },
                ensure_ascii=False,
            )

        except Exception as e:
            return json.dumps({"error": _err("chunk_query_failed", self._lang, error=str(e))})

    async def _extract_key_concepts(self, params: Dict[str, Any]) -> str:
        """Extract key concepts/facts from document chunks and generate probe Q&A."""
        doc_name = params["doc_name"]
        count = min(params.get("count", 20), 50)

        # Look the chunks up by doc_id.
        doc_id = params.get("doc_id")
        chunks = params.get("document_chunks")  # direct input is also allowed

        if doc_id and not chunks:
            chunk_result = await self._retrieve_document_chunks({"doc_id": doc_id, "sample_count": 30})
            try:
                chunk_data = json.loads(chunk_result)
                chunks = [c["content"] for c in chunk_data.get("chunks", []) if c.get("content")]
            except Exception:
                chunks = []

        if not chunks:
            return json.dumps({"error": _err("chunks_empty", self._lang)})

        # Combine chunks for context (limit to avoid token overflow)
        combined_text = "\n\n---\n\n".join(chunks[:30])
        if len(combined_text) > 30000:
            combined_text = combined_text[:30000] + "\n... [truncated]"

        try:
            # A category is only judged once it holds MIN_CATEGORY_PROBES probes,
            # so ask for few enough categories that each one clears that bar.
            n_categories = max(1, min(6, count // max(2 * _MIN_CATEGORY_PROBES, 1)))
            prompt = (
                f"The following is the content of the document '{doc_name}'.\n\n"
                f"{combined_text}\n\n"
                f"---\n\n"
                f"Extract {count} key concepts, facts or pieces of domain knowledge from it.\n"
                f"For each one give:\n"
                f"1. a specific question that tests the concept (question)\n"
                f"2. the correct answer grounded in the document (ground_truth)\n"
                f"3. a sub-category (category)\n\n"
                f"Decide on exactly {n_categories} sub-categories for the whole set, then spread the "
                f"{count} questions evenly across them and reuse those names verbatim. "
                f"Do not invent a category per question — a category holding one question cannot be judged.\n"
                f"Cover a range of difficulty.\n"
                f"Write questions that can be checked against facts, not opinions.\n\n"
                f"Answer with a JSON array only:\n"
                f'[{{"question": "...", "ground_truth": "...", "category": "..."}}]'
            )

            text = await self.complete_text(prompt, max_tokens=8192)

            import re

            # \[\s*\{ ... \}\s*\] — requires array-of-objects pattern to avoid
            # matching stray [bracket text] in qwen3 prose output.
            json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
            if json_match:
                # strict=False: allow literal control chars (e.g. bare \n) inside
                # qwen3 string values — they are still valid semantically.
                probes = json.JSONDecoder(strict=False).decode(json_match.group())
                # Collect categories
                categories = list(set(p.get("category", "general") for p in probes))
                # Cache doc_name for generate_boundary_map.
                self._last_probe_doc = doc_name
                return json.dumps(
                    {
                        "probes": probes[:count],
                        "count": len(probes[:count]),
                        "categories": categories,
                        "doc_name": doc_name,
                        "_next_step": self._rt("next_run_probes"),
                    },
                    ensure_ascii=False,
                )

            return json.dumps({"error": _err("concept_parse_failed", self._lang), "raw": text[:500]})

        except Exception as e:
            return json.dumps({"error": _err("concept_extract_failed", self._lang, error=str(e))})

    async def _run_knowledge_probes(self, params: Dict[str, Any]) -> str:
        """Dispatch the probes to the worker and poll until they finish."""
        probes = params.get("probes") or params.get("probe_questions") or []
        model_name = params.get("model_name", "")

        if not probes:
            return json.dumps({"error": _err("no_probe_questions", self._lang)})
        if not model_name:
            return json.dumps({"error": _err("model_name_empty", self._lang)})

        # ── Step 1: dispatch to the worker ──
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                dispatch_res = await client.post(
                    f"{INTERNAL_API_URL}/api/hf/probe",
                    json={"model_name": model_name, "probes": probes},
                    headers=_internal_headers(),
                )
                if dispatch_res.status_code != 200:
                    return json.dumps(
                        {
                            "error": _err(
                                "probe_dispatch_http_failed",
                                self._lang,
                                status=dispatch_res.status_code,
                                detail=dispatch_res.text[:200],
                            ),
                        },
                        ensure_ascii=False,
                    )
                job_id = dispatch_res.json().get("job_id", "")
                if not job_id:
                    return json.dumps({"error": _err("probe_no_job_id", self._lang)})
        except Exception as e:
            return json.dumps({"error": _err("probe_dispatch_failed", self._lang, error=str(e))}, ensure_ascii=False)

        logger.info(f"[KBD] probe dispatched: job={job_id} model={model_name} probes={len(probes)}")

        # ── Step 2: poll, absorbing both queue wait and worker runtime ──
        # An empty queue runs in tens of seconds; a busy one waits.
        # The timeout is capped at 1h so KBD finishes inside the 2h train timeout.
        _PROBE_POLL_TIMEOUT = 3600
        _POLL_INTERVAL = 1.0

        result_data: Dict[str, Any] = {}
        # During a long poll uvicorn may close the keepalive connection, and the
        # next poll then fails with RemoteProtocolError. The result is already in
        # the worker job state, so a transient transport error must be retried
        # rather than mistaken for a failed analysis.
        _MAX_POLL_TRANSIENT_ERRORS = 10
        _poll_errors = 0
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                elapsed = 0.0
                last_progress = -1
                while elapsed < _PROBE_POLL_TIMEOUT:
                    try:
                        poll_res = await client.get(
                            f"{INTERNAL_API_URL}/api/hf/probe/{job_id}", headers=_internal_headers()
                        )
                    except httpx.TransportError as e:
                        # Transient disconnect, reset or timeout — retry; the result is in the job state.
                        _poll_errors += 1
                        logger.warning(
                            f"[KBD] probe poll transient error ({_poll_errors}/{_MAX_POLL_TRANSIENT_ERRORS}): {e}"
                        )
                        if _poll_errors > _MAX_POLL_TRANSIENT_ERRORS:
                            return json.dumps(
                                {
                                    "error": _err("probe_poll_retry_exceeded", self._lang, error=e),
                                    "job_id": job_id,
                                },
                                ensure_ascii=False,
                            )
                        await asyncio.sleep(_POLL_INTERVAL)
                        elapsed += _POLL_INTERVAL
                        continue
                    _poll_errors = 0  # reset the retry counter on a good response
                    if poll_res.status_code != 200:
                        await asyncio.sleep(_POLL_INTERVAL)
                        elapsed += _POLL_INTERVAL
                        continue

                    state = poll_res.json()
                    status = state.get("status", "PENDING")
                    progress = state.get("progress", 0)

                    if progress != last_progress:
                        logger.info(f"[KBD] probe progress: job={job_id} {progress}% — {state.get('message', '')}")
                        last_progress = progress

                    if status == "SUCCESS":
                        result_data = state.get("result") or {}
                        break
                    if status == "FAILURE":
                        return json.dumps(
                            {
                                "error": _err("probe_run_failed", self._lang, error=state.get("error", "unknown")),
                                "job_id": job_id,
                            },
                            ensure_ascii=False,
                        )

                    await asyncio.sleep(_POLL_INTERVAL)
                    elapsed += _POLL_INTERVAL
                else:
                    return json.dumps(
                        {
                            "error": _err("probe_poll_timeout", self._lang, seconds=_PROBE_POLL_TIMEOUT),
                            "job_id": job_id,
                        },
                        ensure_ascii=False,
                    )
        except Exception as e:
            return json.dumps(
                {
                    "error": _err("probe_poll_failed", self._lang, error=str(e)),
                    "job_id": job_id,
                },
                ensure_ascii=False,
            )

        # ── Step 3: cache the result (score_probe_responses picks it up) ──
        results = result_data.get("results", [])
        self._last_probe_results = results
        self._last_probe_model = model_name

        return json.dumps(
            {
                "model": model_name,
                "results": results,
                "total": result_data.get("total", len(results)),
                "errors": result_data.get("errors", 0),
                "_next_step": self._rt("next_score_probes_with_results"),
            },
            ensure_ascii=False,
        )

    async def _score_probe_responses(self, params: Dict[str, Any]) -> str:
        """Auto-score probe responses with the LLM judge."""
        # The LLM may pass this in several shapes, so extract it leniently.
        probe_results = params.get("probe_results") or params.get("results") or params.get("probes") or []

        # A dict was passed — take its results array.
        if isinstance(probe_results, dict):
            probe_results = probe_results.get("results") or probe_results.get("probe_results") or []

        if not isinstance(probe_results, list):
            probe_results = []

        probe_results = [r for r in probe_results if isinstance(r, dict) and r.get("question")]

        # No parameters: fall back to the cached probe results.
        if not probe_results and hasattr(self, "_last_probe_results") and self._last_probe_results:
            probe_results = self._last_probe_results
            logger.info(f"Using cached probe results: {len(probe_results)} items")

        if not probe_results:
            return json.dumps({"error": _err("no_probe_results", self._lang)})

        scored = []

        batch_prompt = kbd_judge_prompt(probe_results, self._lang)

        try:
            text = await self.complete_text(batch_prompt, max_tokens=8192)

            import re

            json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)

            if json_match:
                evaluations = json.JSONDecoder(strict=False).decode(json_match.group())
                eval_map = {e.get("index", i): e for i, e in enumerate(evaluations)}

                for i, item in enumerate(probe_results):
                    eval_data = eval_map.get(i, {"verdict": "unknown", "confidence": 0.5, "explanation": "not graded"})
                    scored.append(
                        {
                            "question": item.get("question", ""),
                            "ground_truth": item.get("ground_truth", ""),
                            "model_answer": item.get("model_answer", ""),
                            "category": item.get("category", "general"),
                            "verdict": eval_data.get("verdict", "unknown"),
                            "confidence": eval_data.get("confidence", 0.5),
                            "explanation": eval_data.get("explanation", ""),
                        }
                    )
            else:
                # Fallback: mark all as unknown
                for item in probe_results:
                    scored.append({**item, "verdict": "unknown", "confidence": 0.5, "explanation": "parse failed"})

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            for item in probe_results:
                scored.append({**item, "verdict": "unknown", "confidence": 0.5, "explanation": f"grading error: {str(e)}"})

        # Cache it for generate_boundary_map.
        self._last_scored_results = scored

        # Cache model_name and doc_name too, in case a small model omits them.
        for item in scored:
            if item.get("category"):
                break
        # Cache the model_name passed by run_knowledge_probes.
        if hasattr(self, "_last_probe_model"):
            model = self._last_probe_model
        else:
            model = "unknown"
        if hasattr(self, "_last_probe_doc"):
            doc = self._last_probe_doc
        else:
            doc = "unknown"

        return json.dumps(
            {
                "scored_results": scored,
                "total": len(scored),
                "_next_step": self._rt("next_boundary_map", model=model, doc=doc),
            },
            ensure_ascii=False,
        )

    def _generate_boundary_map(self, params: Dict[str, Any]) -> str:
        """Generate knowledge boundary map + RAG/FT/Hybrid path recommendation."""
        scored_results = (
            params.get("scored_results")
            or params.get("results")
            or params.get("scored")
            or params.get("probe_results")
            or []
        )
        model_name = params.get("model_name") or getattr(self, "_last_probe_model", "unknown") or "unknown"
        doc_name = params.get("doc_name") or getattr(self, "_last_probe_doc", "unknown") or "unknown"

        # A dict was passed — take its inner array.
        if isinstance(scored_results, dict):
            scored_results = scored_results.get("scored_results") or scored_results.get("results") or []

        if not isinstance(scored_results, list):
            scored_results = []

        scored_results = [r for r in scored_results if isinstance(r, dict) and r.get("verdict")]

        # No parameters: fall back to the cached scored results.
        if not scored_results and hasattr(self, "_last_scored_results") and self._last_scored_results:
            scored_results = self._last_scored_results
            logger.info(f"Using cached scored results: {len(scored_results)} items")

        if not scored_results:
            return json.dumps({"error": _err("no_scored_results", self._lang)})

        total = len(scored_results)

        # Count verdicts
        correct = sum(1 for r in scored_results if r.get("verdict") in ("correct",))
        partially_correct = sum(1 for r in scored_results if r.get("verdict") == "partially_correct")
        hallucinated = sum(1 for r in scored_results if r.get("verdict") == "hallucinated")
        wrong = sum(1 for r in scored_results if r.get("verdict") == "wrong")
        abstained = sum(1 for r in scored_results if r.get("verdict") == "abstained")
        unknown = total - correct - partially_correct - hallucinated - wrong - abstained

        # Knowledge coverage: correct + partial*0.5
        knowledge_coverage = (correct + partially_correct * 0.5) / max(total, 1)
        knowledge_pct = round(knowledge_coverage * 100, 1)

        # Category breakdown
        categories: Dict[str, Dict] = {}
        for r in scored_results:
            cat = r.get("category", "general")
            if cat not in categories:
                categories[cat] = {
                    "total": 0,
                    "correct": 0,
                    "partial": 0,
                    "hallucinated": 0,
                    "wrong": 0,
                    "abstained": 0,
                }
            categories[cat]["total"] += 1
            v = r.get("verdict", "unknown")
            if v == "correct":
                categories[cat]["correct"] += 1
            elif v == "partially_correct":
                categories[cat]["partial"] += 1
            elif v == "hallucinated":
                categories[cat]["hallucinated"] += 1
            elif v == "wrong":
                categories[cat]["wrong"] += 1
            elif v == "abstained":
                categories[cat]["abstained"] += 1

        category_map = []
        for cat, stats in sorted(categories.items()):
            cat_coverage = (stats["correct"] + stats["partial"] * 0.5) / max(stats["total"], 1)
            category_map.append(
                {
                    "category": cat,
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "partially_correct": stats["partial"],
                    "hallucinated": stats["hallucinated"],
                    "wrong": stats["wrong"],
                    "abstained": stats["abstained"],
                    "coverage": round(cat_coverage * 100, 1),
                    "status": _category_status(cat_coverage, stats["total"]),
                }
            )

        _base_rec = _verdict_base_path(knowledge_pct)
        if _base_rec == "RAG":
            recommendation_detail = (
                f"The model already holds {knowledge_pct}% of this knowledge. "
                f"Retrieval alone is enough. "
                f"Indexing the documents covers the remaining {round(100 - knowledge_pct, 1)}%."
            )
        elif _base_rec == "Fine-Tuning":
            recommendation_detail = (
                f"Knowledge coverage is low at {knowledge_pct}%. "
                f"Fine-tuning is needed to teach the domain knowledge directly. "
                f"Retrieval alone cannot make up for missing foundations."
            )
        else:
            recommendation_detail = (
                f"Knowledge coverage is moderate at {knowledge_pct}%. "
                f"A hybrid of retrieval and a light fine-tune (LoRA) is recommended: "
                f"LoRA for the base domain knowledge, retrieval for the details."
            )

        # Hallucination risk assessment
        hallucination_rate = round(hallucinated / max(total, 1) * 100, 1)

        recommendation, _downgrade = _resolve_verdict_path(knowledge_pct, hallucination_rate)
        if _downgrade == "hallucination_high":
            recommendation_detail += (
                f" The hallucination rate of {hallucination_rate}% makes retrieval alone risky — adjusted to hybrid."
            )
        elif _downgrade == "hallucination_very_high":
            recommendation_detail += f" The hallucination rate of {hallucination_rate}% is very high — adjusted to fine-tuning."

        if hallucination_rate > 20:
            hallucination_warning = (
                f"The hallucination rate is high at {hallucination_rate}%. "
                f"The model tends to answer as if it knows what it does not, which makes fine-tuning more important."
            )
        else:
            hallucination_warning = None

        _weak_cats = [c["category"] for c in category_map if c["status"] == _STATUS_UNKNOWN]
        _boundary_cats = [c["category"] for c in category_map if c["status"] == _STATUS_BOUNDARY]
        _insufficient_cats = [c["category"] for c in category_map if c["status"] == _STATUS_INSUFFICIENT]
        _no_category_judged = bool(_insufficient_cats) and not (_weak_cats or _boundary_cats)
        if _no_category_judged:
            logger.warning(
                f"[KBD] no focus categories — not one category could be judged "
                f"({len(_insufficient_cats)} had too few samples, under {_MIN_CATEGORY_PROBES} probes)."
            )

        # The full record: without the answer and the judge's reasoning a verdict
        # cannot be rechecked, and the answers live only in the worker job state.
        per_question = [
            {
                "question": r.get("question", ""),
                "ground_truth": r.get("ground_truth", ""),
                "category": r.get("category", ""),
                "model_answer": r.get("model_answer", ""),
                "verdict": r.get("verdict", ""),
                "confidence": r.get("confidence"),
                "explanation": r.get("explanation", ""),
            }
            for r in scored_results
            if r.get("question") and r.get("ground_truth")
        ]

        result = {
            "model": model_name,
            "document": doc_name,
            "total_probes": total,
            "per_question": per_question,
            "knowledge_boundary": {
                "knowledge_coverage_pct": knowledge_pct,
                "correct": correct,
                "partially_correct": partially_correct,
                "hallucinated": hallucinated,
                "wrong": wrong,
                "abstained": abstained,
                "unknown": unknown,
            },
            "category_map": category_map,
            # Said plainly, because an empty weak-category list plus an
            # instruction to report one is what makes a model invent names.
            "category_note": (
                f"No category could be judged: every one held fewer than {_MIN_CATEGORY_PROBES} probes. "
                f"Do not report weak categories for this run."
                if _no_category_judged
                else ""
            ),
            "recommendation": {
                "path": recommendation,
                "detail": recommendation_detail,
                "hallucination_rate_pct": hallucination_rate,
                "hallucination_warning": hallucination_warning,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "__chain_signal__": {
                "stage_completed": "kbd",
                "metadata": {
                    "recommendation": recommendation,
                    "knowledge_coverage_pct": knowledge_pct,
                    "hallucination_rate_pct": hallucination_rate,
                    "has_training_data": self._has_training_data_for(doc_name),
                    "doc_name": doc_name,
                    "model_name": model_name,
                    "weak_categories": _weak_cats,
                    "boundary_categories": _boundary_cats,
                    "insufficient_categories": _insufficient_cats,
                },
            },
        }

        # Save result
        self._save_kbd_result(result)

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _has_training_data_for(doc_name: str) -> bool:
        """Whether training data exists for the document being analysed.

        Old data generated from a different document returns False, which is what
        keeps the upload-then-KBD-then-generate pipeline correct.
        """
        doc_stem = Path(doc_name).stem.lower() if doc_name else ""
        for d in iter_dataset_dirs():
            # Does the folder name contain the document name?
            folder_lower = d.name.lower()
            if doc_stem and doc_stem in folder_lower:
                if (d / "qa_dataset.json").exists() or (d / "qa_dataset_dlmax.json").exists():
                    return True
            # Check the source document in the generated dataset metadata.
            meta_path = d / "generation_meta.json"
            if meta_path.exists():
                try:
                    import json as _j

                    meta = _j.loads(meta_path.read_text(encoding="utf-8"))
                    source = meta.get("source_document", "").lower()
                    if doc_stem and doc_stem in source:
                        if (d / "qa_dataset.json").exists() or (d / "qa_dataset_dlmax.json").exists():
                            return True
                except Exception:
                    pass
        return False

    def _save_kbd_result(self, result: Dict):
        """Save KBD result to file."""
        try:
            kbd_dir = self.resolve_path("kbd_results")
            kbd_dir.mkdir(parents=True, exist_ok=True)
            filename = f"kbd_{time.strftime('%Y%m%d_%H%M%S')}.json"
            (kbd_dir / filename).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"KBD result saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save KBD result: {e}")

    async def _list_local_models(self) -> str:
        """Base models installed under the models directory. KBD probes run
        them through `/api/hf/probe`."""
        try:
            # The resolved storage root first, then the project-local fallback.
            _candidates = [Path(os.getenv("STORAGE_BASE_PATH", "storage")) / "models"]
            _candidates.append(Path(__file__).resolve().parents[3] / "storage" / "models")
            base_dir = next((p for p in _candidates if p.exists()), _candidates[0])

            models: List[Dict[str, Any]] = []
            if base_dir.exists():
                for entry in sorted(base_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    if not (entry / "config.json").exists():
                        continue
                    # LoRA adapters and training outputs are not base models.
                    if (entry / "adapter_config.json").exists():
                        continue
                    if (entry / "training_params.json").exists():
                        continue
                    models.append({"name": entry.name})

            if self.autonomy == "guided":
                hint = self._rt("next_concepts_guided")
            else:
                hint = self._rt("next_concepts_auto")
            return json.dumps(
                {
                    "models": models,
                    "count": len(models),
                    "_source": "hf_local",
                    "_next_step": hint,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": _err("model_list_failed", self._lang, error=str(e))})

    def _get_kbd_history(self) -> str:
        """Previous KBD analysis result list."""
        kbd_dir = self.resolve_path("kbd_results")
        results = []
        if kbd_dir.exists():
            for f in sorted(kbd_dir.glob("kbd_*.json"), reverse=True):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    results.append(
                        {
                            "filename": f.name,
                            "model": data.get("model"),
                            "document": data.get("document"),
                            "knowledge_coverage_pct": data.get("knowledge_boundary", {}).get("knowledge_coverage_pct"),
                            "recommendation": data.get("recommendation", {}).get("path"),
                            "timestamp": data.get("timestamp"),
                        }
                    )
                except Exception:
                    continue
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
