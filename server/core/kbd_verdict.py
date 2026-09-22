"""KBD path verdict — one implementation for every caller.

Coverage is a "how many were right" ratio and cannot tell "I do not know" apart
from a fabricated answer, so a high hallucination rate downgrades the verdict: a
model that hallucinates does not know its own boundary, and retrieval alone will
not fix that.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Path names are a machine contract: pipeline routing compares these values
# directly, so they are never translated or changed.
PATH_RAG = "RAG"
PATH_HYBRID = "Hybrid"
PATH_FINE_TUNING = "Fine-Tuning"

#: Coverage (%) at or above which retrieval alone is enough.
COVERAGE_RAG_MIN = 70.0
#: Coverage (%) at or below which adaptation (training) is needed.
COVERAGE_FT_MAX = 50.0

#: Hallucination rate (%) that pulls an RAG verdict down to Hybrid.
HALLUCINATION_DOWNGRADE_RAG = 15.0
#: Hallucination rate (%) that pulls a Hybrid verdict down to Fine-Tuning.
HALLUCINATION_DOWNGRADE_HYBRID = 25.0


def base_path(coverage_pct: float) -> str:
    """First-pass path, from coverage alone."""
    if coverage_pct >= COVERAGE_RAG_MIN:
        return PATH_RAG
    if coverage_pct <= COVERAGE_FT_MAX:
        return PATH_FINE_TUNING
    return PATH_HYBRID


def resolve_path(coverage_pct: float, hallucination_pct: Optional[float] = None) -> Tuple[str, str]:
    """KBD path verdict as ``(path, adjustment reason)``.

    An empty reason means "no adjustment". The caller appends it to the detail
    text so the user sees why the verdict was lowered — never a silent downgrade.

    The adjustment is an ``elif`` on purpose: an RAG verdict with a hallucination
    rate above 25% still only drops to Hybrid, one step at a time. Making it two
    steps would change the verdict of existing auto runs.

    Args:
        coverage_pct: knowledge coverage, 0 to 100.
        hallucination_pct: hallucination rate, 0 to 100. ``None`` skips the
            adjustment — filling an unknown with 0 would read as "no hallucination".
    """
    path = base_path(coverage_pct)
    if hallucination_pct is None:
        return path, ""

    if hallucination_pct > HALLUCINATION_DOWNGRADE_RAG and path == PATH_RAG:
        return PATH_HYBRID, "hallucination_high"
    if hallucination_pct > HALLUCINATION_DOWNGRADE_HYBRID and path == PATH_HYBRID:
        return PATH_FINE_TUNING, "hallucination_very_high"
    return path, ""


def needs_adaptation(path: str) -> bool:
    """Whether this path requires adaptation (fine-tuning).

    Must agree with ``pipeline_routing.needs_adaptation``, which substring-matches
    free-form strings while this takes only the canonical values from this module.
    A test pins the two together.
    """
    return path in (PATH_FINE_TUNING, PATH_HYBRID)


CATEGORY_KNOWN_MIN = 0.7
#: Lower bound for "boundary". Below this the category counts as unknown.
CATEGORY_BOUNDARY_MIN = 0.5

STATUS_KNOWN = "known"
STATUS_BOUNDARY = "boundary"
STATUS_UNKNOWN = "unknown"
STATUS_INSUFFICIENT = "insufficient"

MIN_CATEGORY_PROBES = 2


def category_status(coverage: float, total_probes: Optional[int] = None) -> str:
    """Per-category knowledge status. ``coverage`` is a 0 to 1 ratio.

    The vocabulary is unified here. Previously one caller used
    within/boundary/outside and another known/boundary/unknown for the same
    states, and since consumers compare strings, a split name silently broke filters.

    Args:
        coverage: ratio from 0 to 1.
        total_probes: probes actually run for this category. ``None`` skips the
            check, preserving the behaviour of callers that do not know the count
            — an unknown count is not declared "insufficient". Callers that know it must pass it.

    Returns:
        ``known`` | ``boundary`` | ``unknown`` | ``insufficient``.

    """
    if total_probes is not None and total_probes < MIN_CATEGORY_PROBES:
        return STATUS_INSUFFICIENT
    if coverage >= CATEGORY_KNOWN_MIN:
        return STATUS_KNOWN
    if coverage >= CATEGORY_BOUNDARY_MIN:
        return STATUS_BOUNDARY
    return STATUS_UNKNOWN


