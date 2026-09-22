"""Which files in a dataset folder are the actual training data.

**Single source of truth.** Training (``load_corpus_records``) and the agent's
dataset count (``dispatcher._gather_training_context``) must use the same rule,
or the sample count on screen differs from the one that actually trains.
"""

from __future__ import annotations

from typing import List, Tuple

# Metadata, statistics and lineage files, matched on the lowercased stem.
META_PATTERNS = (
    "_stats",
    "_params",
    "_config",
    "generation_params",
    "preprocessing_stats",
    "deployment_config",
    "training_params",
    "provenance",  # provenance.json — lineage sidecar, not data
)


def _stem(name: str) -> str:
    """Lowercased stem with one extension removed."""
    base = name.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base.lower()


def is_data_file(name: str) -> bool:
    """Whether this is real training data. ``.json`` and ``.jsonl`` only."""
    lowered = name.lower()
    if not (lowered.endswith(".json") or lowered.endswith(".jsonl")):
        return False
    stem = _stem(name)
    return not any(pat in stem for pat in META_PATTERNS)


TIERS: Tuple[Tuple[str, ...], ...] = (
    ("_twist",),
    ("_preprocessed", "_refined"),
)

REFINED_SUFFIXES: Tuple[str, ...] = tuple(suffix for tier in TIERS for suffix in tier)


#: Name of an augmented file. Producer and consumer must read this constant,
#: never a copy of the string.
TWIST_FILE = "qa_dataset_twist.json"

PREPROCESSED_FILE = "qa_dataset_preprocessed.json"


def is_refined(name: str) -> bool:
    """Whether this output replaces the original. ``*_refined_stats`` is already META."""
    return _stem(name).endswith(REFINED_SUFFIXES)


def tier_of(name: str) -> int:
    """Precedence of ``name``: lower wins; a non-replacement gets the original tier.

    The original tier is ``len(TIERS)``, so adding a tier keeps it at the bottom.
    """
    stem = _stem(name)
    for index, suffixes in enumerate(TIERS):
        if stem.endswith(suffixes):
            return index
    return len(TIERS)


def select_data_files(filenames: List[str]) -> List[str]:
    """File names in a folder to the ones training actually reads, sorted.

    The same rule as ``load_corpus_records``. An empty list means no training data.

    Only the highest tier survives: picking the original alongside the
    preprocessed one would revive filtered records and train survivors twice.
    """
    data = sorted(n for n in filenames if isinstance(n, str) and is_data_file(n))
    if not data:
        return []
    best = min(tier_of(n) for n in data)
    data = [n for n in data if tier_of(n) == best]
        # Drop the .jsonl when a .json with the same stem exists.
    json_stems = {_stem(n) for n in data if n.lower().endswith(".json")}
    return [n for n in data if not (n.lower().endswith(".jsonl") and _stem(n) in json_stems)]


def count_samples(path) -> int:
    """Training samples in a dataset file. 0 when it cannot be read.

    JSON holds a list, JSONL one record per line. Counting only the JSON shape
    reports 0 for every JSONL upload, which silently skips the small-data
    adjustment, so both are handled in one place.
    """
    import json
    from pathlib import Path

    if path is None:
        return 0
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return 0
    if p.suffix.lower() == ".jsonl":
        return sum(1 for line in text.splitlines() if line.strip())
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return 0
    return len(payload) if isinstance(payload, list) else 0
