"""List the datasets a training tool may choose from.

The listing must use the same rule as the rest of training — every root in
``dataset_paths``, and ``corpus_files.select_data_files`` to pick the file —
or the LLM chooses from a different set than the one that will be trained.

Depends on the standard library and server.core only, so it unit-tests in a CI venv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from server.core.dataset_paths import dataset_roots
from server.core.lang import EN

DATASET_ROOTS = tuple(str(root) for root in dataset_roots())


def order_datasets(
    datasets: Sequence[Dict[str, Any]],
    requested: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Put the dataset the pipeline chose first.

    **With a choice made, the LLM does not get to pick.** List order is itself a
    hint, so the chosen one goes first and is marked. A name not in the list
    (stale or misspelled) leaves the order untouched — nothing is invented.
    """
    name = (requested or "").strip()
    if not name:
        return list(datasets)
    picked = [d for d in datasets if str(d.get("name")) == name]
    if not picked:
        return list(datasets)
    rest = [d for d in datasets if str(d.get("name")) != name]
    for d in picked:
        d["requested"] = True
    return picked + rest


_HINT: Dict[str, str] = {
    EN: "The pipeline prepared '{name}' for this run — use it. Do not pick a different dataset from this list.",
}

#: Nothing to train on. This edition has no QA generation, so a dataset only
#: arrives by upload — say that instead of letting the LLM invent a name.
_HINT_EMPTY: Dict[str, str] = {
    EN: (
        "No training dataset is available. Do not call start_training_job and do not "
        "invent a dataset name. Ask the user to upload a QA dataset (JSON or JSONL) "
        "on the data page, then run this tool again."
    ),
}

#: Datasets exist, but none holds a usable data file.
_HINT_NO_DATA: Dict[str, str] = {
    EN: (
        "No dataset here holds training data. Ask the user to upload a QA dataset "
        "(JSON or JSONL) on the data page."
    ),
}


def listing_hint(
    datasets: Sequence[Dict[str, Any]],
    requested: Optional[str] = None,
    lang: object = None,
) -> str:
    """Guidance shown alongside the list."""
    if not datasets:
        return _HINT_EMPTY[EN]
    name = (requested or "").strip()
    if name and any(str(d.get("name")) == name for d in datasets):
        return _HINT[EN].format(name=name)
    # Only on explicit evidence: a caller that does not report has_data at all
    # is not claiming the datasets are empty.
    flags = [d.get("has_data") for d in datasets]
    if False in flags and True not in flags:
        return _HINT_NO_DATA[EN]
    return ""
