"""KBD prompts, kept in one place because wording is part of the measurement.

These prompts produce the coverage number that decides RAG versus fine-tuning,
so a pack must keep the same keys, signatures, JSON schema and verdict labels
for results to stay comparable. This build ships English only.
"""

from __future__ import annotations

from typing import Callable, Dict

from server.core.lang import EN


def _kbd_doc_probe_en(doc_name: str, combined: str, count: int) -> str:
    return (
        f"The following is the content of the document '{doc_name}'.\n\n"
        f"{combined}\n\n"
        f"---\n\n"
        f"Extract {count} key concepts, facts, or pieces of domain knowledge from the document above.\n"
        f"For each item provide:\n"
        f"1. A specific question that tests the concept (question)\n"
        f"2. An accurate answer grounded in the document (ground_truth)\n"
        f"3. A sub-category (category)\n\n"
        f"Cover a range of difficulties and categories.\n"
        f"Write specific questions whose answers can be verified against the document.\n\n"
        f"Answer with a JSON array only:\n"
        f'[{{"question": "...", "ground_truth": "...", "category": "..."}}]'
    )


def _kbd_domain_probe_en(domain: str, count: int, difficulty: str) -> str:
    return (
        f"Generate {count} knowledge-boundary-detection (KBD) probe questions for the "
        f"'{domain}' domain.\n\n"
        f"Difficulty: {difficulty} (easy/medium/hard/mixed)\n\n"
        f"Every question must have one accurate answer.\n"
        f"Cover a range of sub-categories.\n\n"
        f"Answer in JSON array format:\n"
        f'[{{"question": "question", "ground_truth": "answer", "category": "sub-category"}}]'
    )


def _kbd_answer_en(question: str) -> str:
    """Prompt that asks the target model to answer a probe question."""
    return f"Answer the following question concisely and accurately.\n\nQuestion: {question}\n\nAnswer:"


def _kbd_answer_fallback_en(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _kbd_judge_en(question: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"Evaluate the model's answer to the following question.\n\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Model answer: {model_answer}\n\n"
        f"Answer in this JSON format only:\n"
        f'{{"verdict": "...", "confidence": 0.0~1.0, "explanation": "one-line explanation", "fabricated": true/false}}\n\n'
        f"verdict criteria:\n"
        f"- correct: semantically matches the ground truth\n"
        f"- hallucinated: confidently states as fact something absent from the ground truth\n"
        f"- abstained: expresses uncertainty, e.g. 'I don't know', 'I'm not sure'\n"
        f"- partially_correct: only partly right\n"
        f"- wrong: simply incorrect (without confidence)\n\n"
        f"confidence: how confident the model's answer sounds (higher = more confident)\n"
        f"fabricated: true if the model invented a new fact absent from the ground truth"
    )


# ──────────────────────────────────────────────────────────
# The packs. Every language must expose the same keys and signatures.
# ──────────────────────────────────────────────────────────

_EN: Dict[str, Callable] = {
    "kbd_doc_probe": _kbd_doc_probe_en,
    "kbd_domain_probe": _kbd_domain_probe_en,
    "kbd_judge": _kbd_judge_en,
    "kbd_answer": _kbd_answer_en,
    "kbd_answer_fallback": _kbd_answer_fallback_en,
}

PACKS: Dict[str, Dict[str, Callable]] = {EN: _EN}


def pack(lang: str | None = None) -> Dict[str, Callable]:
    """Prompt pack for the execution language."""
    return _EN


def prompt(name: str, lang: str | None = None, /, *args, **kwargs):
    """Pick one from the pack and render it, so callers stay short."""
    return pack(lang)[name](*args, **kwargs)
