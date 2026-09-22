"""fastapi-free pure module, so it runs without the web stack."""

import json

from server.core.classify_dataset_format import (
    PREFERENCE,
    SFT,
    UNKNOWN,
    classify_dataset_format,
    resolve_upload_target,
    write_dataset_format,
)
from server.core.dataset_files import read_dataset_format


def _b(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _jsonl(rows) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)).encode("utf-8")


# --- SFT --------------------------------------------------------------------


def test_sft_json_list():
    data = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    assert classify_dataset_format(_b(data), ".json") == SFT


def test_sft_jsonl():
    data = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    assert classify_dataset_format(_jsonl(data), ".jsonl") == SFT


def test_sft_single_dict_json():
    assert classify_dataset_format(_b({"question": "q", "answer": "a"}), ".json") == SFT


def test_sft_extra_keys_ok():
    # Extra keys (instruction, cot_reasoning) do not change the required-key decision.
    row = {"instruction": "x", "question": "q", "cot_reasoning": "...", "answer": "a"}
    assert classify_dataset_format(_b([row]), ".json") == SFT


def test_sft_wrapper_dict_preserved():
    # The wrapper form ({"qa_pairs":[...]}) still works.
    data = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    assert classify_dataset_format(_b(data), ".json") == SFT


# --- Preference -------------------------------------------------------------


def test_preference_json_list():
    data = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
    assert classify_dataset_format(_b(data), ".json") == PREFERENCE


def test_preference_jsonl():
    data = [
        {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
        {"prompt": "p2", "chosen": "c2", "rejected": "r2"},
    ]
    assert classify_dataset_format(_jsonl(data), ".jsonl") == PREFERENCE


def test_preference_wins_over_sft_when_both_present():
    row = {"question": "q", "answer": "a", "prompt": "p", "chosen": "c", "rejected": "r"}
    assert classify_dataset_format(_b([row]), ".json") == PREFERENCE


# --- Unknown ----------------------------------------------------------------


def test_partial_preference_keys_unknown():
    # prompt and chosen without rejected is unknown (strict).
    assert classify_dataset_format(_b([{"prompt": "p", "chosen": "c"}]), ".json") == UNKNOWN


def test_nonstandard_keys_unknown():
    assert classify_dataset_format(_b([{"q": "x", "a": "y"}]), ".json") == UNKNOWN


def test_empty_file_unknown():
    assert classify_dataset_format(b"", ".json") == UNKNOWN
    assert classify_dataset_format(b"", ".jsonl") == UNKNOWN


def test_blank_jsonl_unknown():
    assert classify_dataset_format(b"\n\n   \n", ".jsonl") == UNKNOWN


def test_malformed_json_unknown():
    assert classify_dataset_format(b"{not valid json", ".json") == UNKNOWN
    assert classify_dataset_format(b"{broken\n", ".jsonl") == UNKNOWN


def test_bad_encoding_unknown():
    # Invalid UTF-8 bytes.
    assert classify_dataset_format(b"\xff\xfe\x00bad", ".json") == UNKNOWN
    assert classify_dataset_format(b"\xff\xfe\x00bad", ".jsonl") == UNKNOWN


def test_unsupported_ext_unknown():
    assert classify_dataset_format(_b([{"question": "q", "answer": "a"}]), ".csv") == UNKNOWN
    assert classify_dataset_format(b"", "") == UNKNOWN


def test_empty_list_unknown():
    assert classify_dataset_format(_b([]), ".json") == UNKNOWN


# --- Large .jsonl: only first line parsed -----------------------------------


def test_large_jsonl_reads_first_line_only():
    # The first line is valid sft and the rest is broken JSON, so only the first line matters.
    first = json.dumps({"question": "q", "answer": "a"})
    garbage = "X" * (3 * 1024 * 1024)  # 3MB of junk that would error if parsed
    content = (first + "\n" + garbage).encode("utf-8")
    assert classify_dataset_format(content, ".jsonl") == SFT


# --- resolve_upload_target --------------------------------------------------


def test_resolve_target_preference_forces_training():
    assert resolve_upload_target("processed", PREFERENCE) == "training"
    assert resolve_upload_target("training", PREFERENCE) == "training"
    assert resolve_upload_target("raw", PREFERENCE) == "training"


def test_resolve_target_sft_keeps_requested():
    assert resolve_upload_target("processed", SFT) == "processed"
    assert resolve_upload_target("training", SFT) == "training"


def test_resolve_target_none_keeps_requested():
    assert resolve_upload_target("raw", None) == "raw"
    assert resolve_upload_target("processed", None) == "processed"


# --- sidecar write/read roundtrip -------------------------------------------


def test_write_read_format_roundtrip(tmp_path):
    d = tmp_path / "dataset_xxx"
    d.mkdir()
    p = write_dataset_format(d, SFT, source_filename="qa_dataset.json")
    assert p.exists()
    assert p.name == "format.json"
    assert ".koni_meta" in p.parts
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["data_format"] == "sft"
    assert payload["source_filename"] == "qa_dataset.json"
    assert "classified_at" in payload
    assert read_dataset_format(d) == "sft"


def test_write_read_preference(tmp_path):
    d = tmp_path / "dataset_yyy"
    d.mkdir()
    write_dataset_format(d, PREFERENCE)
    assert read_dataset_format(d) == "preference"


def test_read_format_absent_returns_none(tmp_path):
    d = tmp_path / "dataset_zzz"
    d.mkdir()
    assert read_dataset_format(d) is None  # carry-over (pre-C1-a)


def test_read_format_corrupt_returns_none(tmp_path):
    d = tmp_path / "dataset_bad"
    (d / ".koni_meta").mkdir(parents=True)
    (d / ".koni_meta" / "format.json").write_text("{broken", encoding="utf-8")
    assert read_dataset_format(d) is None
