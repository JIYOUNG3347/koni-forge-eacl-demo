"""Pure module, free of torch and fastapi, so it runs in a plain CI venv."""

import server.core.model_match as mm

KNOWN = ["Qwen--Qwen2.5-7B-Instruct", "Qwen--Qwen3-4B", "google--gemma-3-1b-it"]


# ── normalize ───────────────────────────────────────────────
def test_normalize_strips_separators_and_case():
    assert mm.normalize("Qwen--Qwen3-4B") == "qwenqwen34b"
    assert mm.normalize("Qwen/Qwen3-4B") == "qwenqwen34b"
    assert mm.normalize("  qwen_qwen3 4b  ") == "qwenqwen34b"


def test_normalize_handles_dots():
    assert mm.normalize("Qwen--Qwen2.5-7B-Instruct") == "qwenqwen257binstruct"


def test_normalize_non_string():
    assert mm.normalize(None) == ""  # type: ignore[arg-type]
    assert mm.normalize(42) == ""  # type: ignore[arg-type]


def test_repro_2026_07_21_incident():
    assert mm.resolve_model_name("Qwen3-4B", KNOWN) == "Qwen--Qwen3-4B"


def test_repo_id_form_resolves():
    assert mm.resolve_model_name("Qwen/Qwen3-4B", KNOWN) == "Qwen--Qwen3-4B"


def test_loose_spacing_and_case_resolves():
    assert mm.resolve_model_name("qwen3 4b", KNOWN) == "Qwen--Qwen3-4B"
    assert mm.resolve_model_name("QWEN3_4B", KNOWN) == "Qwen--Qwen3-4B"


def test_exact_folder_name_resolves():
    assert mm.resolve_model_name("Qwen--Qwen2.5-7B-Instruct", KNOWN) == "Qwen--Qwen2.5-7B-Instruct"


def test_org_prefixed_other_vendor():
    assert mm.resolve_model_name("gemma-3-1b-it", KNOWN) == "google--gemma-3-1b-it"


# ── Undecidable cases return None (never pick arbitrarily) ──
def test_unknown_model_is_none():
    assert mm.resolve_model_name("llama-3-70b", KNOWN) is None


def test_empty_candidate_is_none():
    assert mm.resolve_model_name("", KNOWN) is None
    assert mm.resolve_model_name("   ", KNOWN) is None


def test_ambiguous_tail_is_none():
    # Two orgs with the same model name are never disambiguated arbitrarily.
    known = ["orgA--Qwen3-4B", "orgB--Qwen3-4B"]
    assert mm.resolve_model_name("Qwen3-4B", known) is None


def test_prefix_only_is_not_a_match():
    # "Qwen" is not a suffix of either Qwen model, so the result is None.
    assert mm.resolve_model_name("Qwen", KNOWN) is None


def test_longer_variant_not_matched():
    # "Qwen3-4B" must not attach to a longer model ending in "-Instruct".
    known = ["Qwen--Qwen3-4B-Instruct"]
    assert mm.resolve_model_name("Qwen3-4B", known) is None


def test_empty_known_list():
    assert mm.resolve_model_name("Qwen3-4B", []) is None


# ── resolve_or_none ─────────────────────────────────────────
def test_resolve_or_none_passthrough_exact():
    assert mm.resolve_or_none("Qwen--Qwen3-4B", KNOWN) == "Qwen--Qwen3-4B"


def test_resolve_or_none_falls_back_to_fuzzy():
    assert mm.resolve_or_none("Qwen3-4B", KNOWN) == "Qwen--Qwen3-4B"


def test_resolve_or_none_unknown():
    assert mm.resolve_or_none("nope", KNOWN) is None


























