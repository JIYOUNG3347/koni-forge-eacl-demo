import gc
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from server.core.config import STORAGE_ROOT
from server.core.prompt_packs import prompt as _kbd_prompt

# --- Environment and shared state ---

# Every root derives from the resolved storage root.
# Root directory holding base model files.
BASE_MODELS_ROOT = str(STORAGE_ROOT / "models")
# Legacy fine-tuning output root (real path: OUTPUTS_ROOT/{user_id}/completed/{job_id}).
TRAINED_OUTPUTS_ROOT = str(STORAGE_ROOT / "outputs" / "completed")
# Per-user output root.
OUTPUTS_ROOT = str(STORAGE_ROOT / "outputs")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache. One model in memory at a time, to keep device memory free.
inference_state: Dict[str, Any] = {
    "model": None,  # loaded model (AutoModelForCausalLM or PeftModel)
    "tokenizer": None,  # loaded tokenizer
    "model_name": None,  # full path of the loaded model, used to skip a reload
}

# Fallback chat template for tokenizers that ship without one.
DEFAULT_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"

# --- Utilities ---


def _fix_tokenizer_config(model_dir: str):
    """transformers >= 4.56: convert an extra_special_tokens list to a dict."""
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        est = config.get("extra_special_tokens")
        if isinstance(est, list):
            config["extra_special_tokens"] = {t: t for t in est}
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Fixed extra_special_tokens list→dict in {config_path}")
    except Exception:
        pass


def check_bf16_capability() -> bool:
    """Whether the current device supports bfloat16.

    Returns:
        True on Ampere or newer (compute capability 8.0+).
    """
    try:
        # Index 0 is the first device visible to this process (already remapped
        # by CUDA_VISIBLE_DEVICES), not a physical topology index, so it needs
        # no configuration. Homogeneous devices are assumed.
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return False


def locate_model_directory(model_name: str) -> str:
    """Find the on-disk path for a model name.

    Search order:
      1. BASE_MODELS_ROOT/{model_name}
      2. TRAINED_OUTPUTS_ROOT/{model_name}  (legacy)
      3. OUTPUTS_ROOT/{user_id}/completed/{model_name}  (real training output)
    """
    # Name candidates: the original plus the flattened HuggingFace download name.
    # download_hf_task replaces "/" in a repo id with "--"
    # (safe_name = model_id.replace("/", "--")), so a repo id such as
    # "Qwen/Qwen2.5-3B-Instruct" must also resolve to the on-disk
    # "Qwen--Qwen2.5-3B-Instruct".
    name_variants = [model_name]
    if "/" in model_name and not os.path.isabs(model_name):
        name_variants.append(model_name.replace("/", "--"))

    candidates = []
    for nm in name_variants:
        candidates.append(os.path.join(BASE_MODELS_ROOT, nm))
        candidates.append(os.path.join(TRAINED_OUTPUTS_ROOT, nm))
    # Also search the per-user completed directory.
    if os.path.isdir(OUTPUTS_ROOT):
        for user_id in os.listdir(OUTPUTS_ROOT):
            for nm in name_variants:
                candidates.append(os.path.join(OUTPUTS_ROOT, user_id, "completed", nm))

    for p in candidates:
        if os.path.isdir(p):
            return p
    raise ValueError(f"Model '{model_name}' not found in any search path. Tried: {candidates[:4]}")


def calc_min_gen_length(input_ids_len: int, floor: int = 24, cap: int = 128) -> int:
    """Minimum number of tokens to generate, scaled to the input length.

    Long inputs should not get a one-word answer.

    Args:
        input_ids_len: input token count
        floor: lower bound (default 24)
        cap: upper bound (default 128)

    Returns:
        The min_new_tokens value.
    """
    # Half the input length, clamped to [floor, cap].
    return max(floor, min(cap, input_ids_len // 2))


# --- Core functions ---


def scan_available_models() -> List[str]:
    """Scan every servable model: base models, fine-tuned models and checkpoints.

    Logic:
        1. Base model folders under BASE_MODELS_ROOT
        2. Job folders under TRAINED_OUTPUTS_ROOT
        3. Final outputs and 'checkpoint-*' subfolders in each job folder
        4. Validity checked by config.json or adapter_config.json

    Returns:
        Sorted model names.
    """
    all_models: List[str] = []

    # Base models.
    if os.path.isdir(BASE_MODELS_ROOT):
        for model_name in os.listdir(BASE_MODELS_ROOT):
            path = os.path.join(BASE_MODELS_ROOT, model_name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                all_models.append(model_name)

    # Fine-tuned models and checkpoints.
    if os.path.isdir(TRAINED_OUTPUTS_ROOT):
        for job_name in os.listdir(TRAINED_OUTPUTS_ROOT):
            job_path = os.path.join(TRAINED_OUTPUTS_ROOT, job_name)
            if not os.path.isdir(job_path):
                continue

            # Final output (a merged model or an adapter).
            if os.path.exists(os.path.join(job_path, "config.json")) or os.path.exists(
                os.path.join(job_path, "adapter_config.json")
            ):
                all_models.append(job_name)

            # Intermediate checkpoints.
            for item in os.listdir(job_path):
                item_path = os.path.join(job_path, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    if os.path.exists(os.path.join(item_path, "config.json")) or os.path.exists(
                        os.path.join(item_path, "adapter_config.json")
                    ):
                        all_models.append(f"{job_name}/{item}")

    return sorted(list(set(all_models)))  # de-duplicate and sort


def release_loaded_model():
    """Release the model and tokenizer held in inference_state.

    Details:
        - drop the Python references
        - empty the device cache
        - force a garbage collection
    """
    global inference_state
    if inference_state["model"] is not None:
        try:
            del inference_state["model"]
            del inference_state["tokenizer"]
        except Exception:
            pass
        inference_state.update({"model": None, "tokenizer": None, "model_name": None})
        logger.info("Model unloaded.")
    torch.cuda.empty_cache()
    gc.collect()


def activate_model(model_name: str):
    """
    Load a model, including a LoRA adapter.


    Logic:
        1. Resolve the path and return early on a cache hit
        2. Unload the previous model to free memory
        3. Detect a LoRA adapter (adapter_config.json present)
        4. For LoRA:
            - read training_params.json to find the base model path
        5. Pick the dtype the hardware supports (bf16 or fp16)
        6. Load the base model and tokenizer
        7. Patch the tokenizer (pad token, chat template)
        8. For LoRA: attach the adapter with PeftModel
        9. Update the global cache

    Args:
        model_name: one of the names returned by scan_available_models
    """
    global inference_state

    # Resolve the path.
    model_path = locate_model_directory(model_name)

    # Already loaded — cache hit.
    if inference_state["model_name"] == model_path and inference_state["model"] is not None:
        logger.info(f"Model already loaded: {model_path}")
        return

    # Unload the previous model to free memory.
    release_loaded_model()

    # Detect LoRA and resolve the base model path.
    base_model_path = model_path
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        # A LoRA adapter needs its base model, recorded in training_params.json.
        params_path = os.path.join(model_path, "training_params.json")
        if not os.path.exists(params_path):
            # Not inside a checkpoint — look in the parent folder.
            parent_params = os.path.join(os.path.dirname(model_path), "training_params.json")
            if os.path.exists(parent_params):
                params_path = parent_params
            else:
                raise FileNotFoundError(f"training_params.json not found for LoRA model in {model_path} or its parent.")

        # Read base_model_name_or_path out of the JSON.
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        base_model_name_or_path = (
            # Current format.
            params.get("model_path")
            or params.get("job_cfg", {}).get("model_name_or_path")
            # Older format (trl 0.x with ModelConfig).
            or params.get("model_args", {}).get("model_name_or_path")
            or params.get("model_name_or_path")
        )
        if not base_model_name_or_path or not os.path.isdir(base_model_name_or_path):
            raise FileNotFoundError(
                f"Base model path '{base_model_name_or_path}' not found. "
                f"training_params.json keys: {list(params.keys())}"
            )
        base_model_path = base_model_name_or_path

    # Pick the dtype: bf16 when supported, otherwise fp16.
    prefer_bf16 = check_bf16_capability()
    dtype = torch.bfloat16 if prefer_bf16 else torch.float16
    logger.info(f"Loading base model from: {base_model_path} (dtype={dtype})")

    # Load the base model and tokenizer (trust_remote_code allows custom architectures).
    # transformers >= 4.56 renamed torch_dtype to dtype.
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        # Older versions.
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    # transformers >= 4.56: an extra_special_tokens list must become a dict.
    _fix_tokenizer_config(base_model_path)
    if is_lora:
        _fix_tokenizer_config(model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Patch pad and eos tokens so generation does not error.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            base_model.resize_token_embeddings(len(tokenizer))

    # Keep pad_token_id in sync on the model config too.
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Supply a chat template for models that ship without one.
    if getattr(tokenizer, "chat_template", None) is None:
        logger.warning("Tokenizer has no chat_template. Applying default template.")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Load and attach the LoRA adapter.
    model = base_model
    if is_lora:
        logger.info(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

    # Update the cache and switch to inference mode.
    inference_state["model"] = model.eval()
    inference_state["tokenizer"] = tokenizer
    inference_state["model_name"] = model_path
    inference_state["is_lora"] = is_lora
    logger.info(f"Successfully loaded model: {model_path}")


def generate_response(
    question: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    lang: object = None,
) -> str:
    """One non-streaming answer, used by the KBD probe."""
    if not inference_state["model"] or not inference_state["tokenizer"]:
        raise RuntimeError("Model is not loaded.")

    tokenizer = inference_state["tokenizer"]
    model = inference_state["model"]

    messages = [{"role": "user", "content": _kbd_prompt("kbd_answer", lang, question)}]
    try:
        chat_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_formatted, return_tensors="pt", padding=True).to(model.device)
    except Exception:
        # The fallback must use the same language, or only models whose template failed answer differently.
        inputs = tokenizer(
            _kbd_prompt("kbd_answer_fallback", lang, question),
            return_tensors="pt",
            padding=True,
        ).to(model.device)

    use_sampling = bool(temperature) and float(temperature) > 0.0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=use_sampling,
            temperature=max(0.01, float(temperature)) if use_sampling else None,
            top_p=0.9 if use_sampling else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# RAG singletons, so the embedding model is not reloaded on every call.
_rag_cache: Dict[str, Any] = {"vs": None, "em": None}


def _get_rag_instances():
    """Return the VectorStore and EmbeddingManager singletons."""
    if _rag_cache["vs"] is None or _rag_cache["em"] is None:
        from modules.rag import EmbeddingManager, VectorStore

        _rag_cache["vs"] = VectorStore()
        _rag_cache["em"] = EmbeddingManager()
        logger.info("RAG instances initialized (singleton)")
    return _rag_cache["vs"], _rag_cache["em"]


# Orchestrator agent singleton used for faithfulness scoring.


def _build_rag_context(
    query: str,
    top_k: int = 3,
    threshold: float = 0.15,
    doc_ids: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Optional[str]:
    """Build a context string from RAG search results.

    Args:
        query: search query
        top_k: number of results
        threshold: minimum similarity
        doc_ids: filter by document id (legacy)
        datasets: filter by dataset (source_folder), preferred
    """
    try:
        vs, em = _get_rag_instances()
        q_emb = em.embed_query(query)

        # Dataset-scoped search when possible, otherwise a full search.
        if datasets and len(datasets) == 1:
            # One dataset: use the ChromaDB where filter.
            results = vs.search_by_dataset(q_emb, datasets[0], top_k=int(top_k) * 2)
        elif datasets and len(datasets) > 1:
            # Several datasets: search everything, then filter.
            results = vs.search(q_emb, top_k=int(top_k) * 4)
            results = [r for r in results if r.get("metadata", {}).get("source_folder") in datasets]
        else:
            results = vs.search(q_emb, top_k=int(top_k) * 2)

        # Debug: log the result scores.
        if results:
            scores_log = ", ".join(f"{r.get('score', 0):.3f}" for r in results[:5])
            logger.info(f"RAG search scores for '{query[:30]}...': [{scores_log}]")
        else:
            logger.warning(f"RAG search returned no results for: {query[:50]}")

        relevant = [r for r in results if r.get("score", 0) >= threshold]
        # Legacy doc_ids filter.
        if doc_ids:
            relevant = [r for r in relevant if r.get("metadata", {}).get("doc_id") in doc_ids]
        relevant = relevant[:top_k]
        if not relevant:
            logger.warning(f"RAG: no results above threshold({threshold}) for: {query[:50]}")
            return None
        chunks = []
        for i, r in enumerate(relevant, 1):
            source = r.get("metadata", {}).get("doc_name", "document")
            chunks.append(f"[Source {i}] ({source}, similarity: {r['score']:.2f})\n{r['content']}")
        return "\n\n".join(chunks)
    except Exception as e:
        import traceback

        logger.warning(f"RAG context retrieval failed: {e}\n{traceback.format_exc()}")
        return None


def inject_rag_context(
    messages: List[Dict[str, str]],
    rag_enabled: bool = False,
    top_k: int = 3,
    doc_ids: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> tuple:
    """Inject the RAG context into a message list.

    Takes the query from the last user message and adds the results as context.

    Returns:
        (modified messages, RAG context string or None)
    """
    if not rag_enabled:
        return messages, None

    # Take the last user message.
    last_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    if not last_user:
        return messages, None

    context = _build_rag_context(last_user, top_k=top_k, doc_ids=doc_ids, datasets=datasets)
    if not context:
        return messages, None

    # Small models tend to ignore system messages, so the context goes
    # directly inside the last user message.
    new_messages = list(messages)
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            original_question = new_messages[i]["content"]
            new_messages[i] = {
                "role": "user",
                "content": (
                    f"[Reference material]\n{context}\n\n"
                    f"[Instruction] Answer only from the reference material above. "
                    f"Do not include anything that is not in it.\n\n"
                    f"[Question] {original_question}"
                ),
            }
            break

    return new_messages, context


def _estimate_response_quality(answer: str) -> float:
    """Estimate answer quality from the text alone, 0.1 to 0.9.

    Looks at lexical diversity and hedging. Used when token probabilities
    """
    if not answer or len(answer) < 10:
        return 0.1
    words = answer.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    uncertainty_phrases = ["i don't know", "not sure", "unclear", "sorry", "cannot determine", "no idea"]
    penalty = 0.15 if any(p in answer for p in uncertainty_phrases) else 0.0
    return max(0.1, min(0.9, unique_ratio * 0.8 + 0.1 - penalty))


def stream_chat_response(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    rag_enabled: bool = False,
    rag_top_k: int = 3,
    rag_doc_ids: Optional[List[str]] = None,
    rag_datasets: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Stream a chat response in SSE format.

    With ``rag_enabled`` the retrieved context is injected into the messages
    before generation; otherwise the model answers on its own.

    Args:
        messages: OpenAI chat-format message list
        temperature: sampling temperature
        max_tokens: maximum tokens to generate
        rag_enabled: whether to use the RAG context
        rag_top_k: number of RAG results
        rag_doc_ids: restrict RAG to these documents (None means all)

    Yields:
        SSE formatted stream ('data: {"token": "..."}\n\n')
    """
    logger.info(f"Generating chat with temperature={temperature}, max_tokens={max_tokens}, rag={rag_enabled}")

    if not inference_state["model"] or not inference_state["tokenizer"]:
        raise RuntimeError("Model is not loaded.")

    tokenizer = inference_state["tokenizer"]
    model = inference_state["model"]
    if rag_enabled:
        messages, rag_context = inject_rag_context(
            messages, rag_enabled=True, top_k=rag_top_k, doc_ids=rag_doc_ids, datasets=rag_datasets
        )
        if rag_context:
            yield f"data: {json.dumps({'rag_context': True, 'sources': len(rag_context.split('[Source')) - 1})}\n\n"
            logger.info(f"RAG context injected: {len(rag_context)} chars")

    # ── Normal streaming generation (fine-tuned, or with RAG injected) ──
    try:
        chat_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info(f"Formatted model input:\n---\n{chat_formatted[:500]}...\n---")
        inputs = tokenizer(chat_formatted, return_tensors="pt", padding=True).to(model.device)
    except Exception as e:
        logger.exception("Error applying chat template / tokenization")
        raise ValueError("Failed to process chat messages with the model's template.") from e

    input_len = inputs.input_ids.shape[-1]
    dynamic_min_new = calc_min_gen_length(input_len)

    use_sampling = bool(temperature) and float(temperature) > 0.0
    safe_temp = max(0.01, float(temperature)) if use_sampling else 0.7

    streamer = TextIteratorStreamer(tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        min_new_tokens=dynamic_min_new,
        do_sample=use_sampling,
        temperature=safe_temp,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )

    import threading

    err_box: Dict[str, Optional[BaseException]] = {"exc": None}

    def _run():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as ex:
            err_box["exc"] = ex
            logger.exception("generate() failed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    try:
        for new_text in streamer:
            if new_text:
                yield f"data: {json.dumps({'token': new_text})}\n\n"
        thread.join()

        if err_box["exc"] is not None:
            yield f"data: {json.dumps({'error': str(err_box['exc'])})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
