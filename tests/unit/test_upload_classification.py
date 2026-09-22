"""How an uploaded QA file is classified and where it lands.

``upload_files`` is a router and cannot be imported in a CI venv, so the pure
pieces it calls are composed here in the same order. Both QA targets share the
corpus root in this edition: there is no generation step for a dataset to wait
in.
"""

import json

from server.core.classify_dataset_format import (
    classify_dataset_format,
    resolve_upload_target,
    write_dataset_format,
)
from server.core.dataset_files import read_dataset_format

#: Upload target -> storage root, matching the router.
ROOTS = {"raw": "raw_corpus", "processed": "corpus", "training": "corpus"}


def _jsonl(rows) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")


def _simulate_upload(tmp_path, content: bytes, ext: str, requested_target: str):
    """The router decision: (effective_target, data_format) plus the sidecar."""
    fmt = classify_dataset_format(content, ext)
    if fmt == "unknown":
        return "REJECTED_400", fmt, None
    effective = resolve_upload_target(requested_target, fmt)
    dest = tmp_path / ROOTS[effective] / "dataset_xxx"
    dest.mkdir(parents=True)
    dest.joinpath(f"qa_dataset{ext}").write_bytes(content)
    if fmt in ("sft", "preference"):
        write_dataset_format(dest, fmt, source_filename=f"qa_dataset{ext}")
    return effective, fmt, dest


def test_an_sft_upload_is_recognised_and_marked(tmp_path):
    effective, fmt, dest = _simulate_upload(
        tmp_path, _jsonl([{"question": "q", "answer": "a"}]), ".jsonl", "processed"
    )
    assert fmt == "sft"
    assert effective == "processed"
    assert "corpus" in dest.parts
    assert read_dataset_format(dest) == "sft"


def test_a_preference_upload_is_forced_to_the_training_target(tmp_path):
    """Its shape decides, not the target the caller asked for."""
    effective, fmt, dest = _simulate_upload(
        tmp_path, _jsonl([{"prompt": "p", "chosen": "c", "rejected": "r"}]), ".jsonl", "processed"
    )
    assert fmt == "preference"
    assert effective == "training"
    assert read_dataset_format(dest) == "preference"


def test_an_unrecognised_shape_is_rejected_and_not_saved(tmp_path):
    effective, fmt, dest = _simulate_upload(tmp_path, _jsonl([{"foo": "bar"}]), ".jsonl", "processed")
    assert fmt == "unknown"
    assert effective == "REJECTED_400"
    assert dest is None
