"""Training-record loading and chat-template rendering."""

from __future__ import annotations

import json
from pathlib import Path


def load_corpus_records(dataset_path: str):
    """
    Load corpus records into a HuggingFace Dataset.
    Reads JSON or JSONL from a file or directory.

    Logic:
        1. Check whether the path is a file or a directory.
        2. For a directory, collect every .json and .jsonl file inside.
        3. Parse by extension (JSON whole-file, JSONL line by line).
        4. Normalise the 'question', 'cot_reasoning' and 'answer' fields.

    Args:
        dataset_path: path to a data file or directory.

    Returns:
        The prepared Dataset, held in memory.

    Raises:
        ValueError: no valid data file in the directory.
        FileNotFoundError: the path does not exist.
    """
    from server.core.corpus_files import select_data_files

    p = Path(dataset_path)
    if p.is_dir():
        all_names = [f.name for f in (list(p.glob("*.json")) + list(p.glob("*.jsonl")))]
        selected = select_data_files(all_names)
        if not selected:
            raise ValueError(f"No valid training data file in directory: {dataset_path}")
        paths = [p / n for n in selected]
        print(f"[INFO] training files selected: {len(paths)} of {len(all_names)} ({[x.name for x in paths]})")
    elif p.is_file():
        paths = [p]
    else:
        raise FileNotFoundError(f"Path does not exist or is not a file or directory: {dataset_path}")

    raw_entries = []
    for path in paths:
        if path.suffix == ".jsonl":
            # JSONL: read line by line, loaded into a list here.
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[WARN] could not parse a line of {path.name}: {e}")
        else:
            # Plain JSON.
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"[WARN] could not parse {path.name}, skipping: {e}")
                continue
            if isinstance(data, dict):
                data = [data]
            raw_entries.extend(data)

    # Normalise the fields (no None, always strings) and drop empty records.
    rows = []
    skipped = 0
    for r in raw_entries:
        question = (r.get("question") or "").strip()
        cot_reasoning = (r.get("cot_reasoning") or "").strip()
        answer = (r.get("answer") or "").strip()
        if not question and not answer:
            skipped += 1
            continue
        rows.append({"question": question, "cot_reasoning": cot_reasoning, "answer": answer})

    if skipped > 0:
        print(f"[WARN] skipped {skipped} records with both question and answer empty.")

    if not rows:
        raise ValueError(f"No valid training data. Check the path: {dataset_path}")

    from datasets import Dataset

    return Dataset.from_list(rows)


def render_chat_sample(
    example,
    tokenizer,
    *,
    tokenize=False,
    add_generation_prompt=False,
    append_eos=True,
    system_prompt=None,
):
    """
    Render one record as a chat-formatted training sample.
    Chain-of-thought content, when present, is wrapped in a <think> tag.

    Args:
        example: one record (question, cot_reasoning, answer).
        tokenizer: tokenizer used to apply the chat template.
        tokenize: True returns input_ids, False returns text.

    Logic:
        1. Build the messages: system (optional), user (question), assistant (CoT + answer).
        2. Apply the tokenizer's `apply_chat_template`.
        3. Fall back to plain concatenation for models without a chat template.

    Returns:
        {'input_ids': ...} or {'text': ...}
    """

    q = (example.get("question") or "").strip()
    a = (example.get("answer") or "").strip()
    cot = (example.get("cot_reasoning") or "").strip()

    # 1. Build the messages in OpenAI chat format.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Wrap chain-of-thought in <think> so the model learns the reasoning step.
    if cot:
        assistant_content = f"<think>\n{cot}\n</think>\n{a}"
    else:
        assistant_content = a

    messages.extend(
        [
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_content},
        ]
    )

    # 2. Prefer the chat template when the tokenizer has one.
    rendered = None
    token_ids = None
    use_chat = hasattr(tokenizer, "chat_template") and bool(tokenizer.chat_template)

    if use_chat:
        try:
            if tokenize:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    return_tensors=None,
                )
            else:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
        except Exception:
            use_chat = False  # template failed — fall back

    # 3. Fallback: plain question and answer text, for base models with no template.
    if not use_chat:
        fallback_text = ""
        if tokenizer.bos_token:
            fallback_text += tokenizer.bos_token

        if cot:
            fallback_text += f"{q}\n<think>\n{cot}\n</think>\n{a}\n"
        else:
            fallback_text += f"{q}\n{a}\n"

        if append_eos and tokenizer.eos_token:
            fallback_text += tokenizer.eos_token

        if tokenize:
            token_ids = tokenizer(fallback_text, add_special_tokens=False)["input_ids"]
        else:
            rendered = fallback_text

    # 4. Return in the shape SFTTrainer expects.
    return {"input_ids": token_ids} if tokenize else {"text": rendered}
