"""Static file cache policy — server/core/static_cache.py."""

import pytest

from server.core.static_cache import (
    IMMUTABLE,
    REVALIDATE,
    cache_control_for,
    is_content_hashed,
)


@pytest.mark.parametrize(
    "name",
    [
        "index-BI0ScPSE.js",
        "index-4izOiXvy.css",
        "landing-illustration-B5E9qtVX.png",
        "guided-hero-illustration-BbX-HKkw.png",
        "jspdf.es.min-itL6hA5L.js",
        "assets/index-Dcr4eBRr.js",
    ],
)
def test_hashed_files_are_immutable(name):
    """A Vite-hashed build artefact is cached permanently."""
    assert is_content_hashed(name) is True
    assert cache_control_for(name) == IMMUTABLE


@pytest.mark.parametrize(
    "name",
    [
        "index.html",
        "favicon.ico",
        "robots.txt",
        "logo.png",
        "vendor-lib.js",  # has a hyphen but no hash (fewer than 8 characters)
        "",
    ],
)
def test_unhashed_files_revalidate(name):
    """Caching a fixed-name file permanently leaves the old page after a deploy."""
    assert is_content_hashed(name) is False
    assert cache_control_for(name) == REVALIDATE


def test_windows_path_separator():
    assert is_content_hashed("assets\\index-BI0ScPSE.js") is True


def test_immutable_value_is_a_year():
    assert "max-age=31536000" in IMMUTABLE
    assert "immutable" in IMMUTABLE
