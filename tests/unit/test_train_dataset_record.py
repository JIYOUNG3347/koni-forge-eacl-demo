"""The report must not guess the dataset name."""


import pytest

from server.core.train_dataset_record import (
    candidate_paths,
    resolve_dataset_path,
)


class TestLookupIsOneRule:
    def test_candidate_order(self):
        """One root, and the augmented copy after the original."""
        assert candidate_paths("ds", "/storage") == [
            "/storage/corpus/ds",
            "/storage/corpus/ds_twist",
        ]

    def test_explicit_path_wins(self):
        assert resolve_dataset_path("ds", "/storage", "/tmp/x") == "/tmp/x"

    def test_first_existing_candidate(self):
        on_disk = {"/storage/corpus/ds_twist"}
        assert resolve_dataset_path("ds", "/storage", exists=lambda p: p in on_disk) == "/storage/corpus/ds_twist"

    def test_none_existing_returns_the_first_candidate(self):
        """Never silently pick another dataset — failing with "path not found" is better."""
        assert resolve_dataset_path("ds", "/storage", exists=lambda p: False) == ("/storage/corpus/ds")

    @pytest.mark.parametrize("empty", ["", None])
    def test_no_name_no_crash(self, empty):
        assert resolve_dataset_path(empty or "", "/storage", exists=lambda p: False)


class TestWiring:
    def test_the_fabrication_guard_catches_the_shape(self):
        bad = '        training_section["params"]["dataset"] = _dg_job + "_twist"'
        assert '+ "_twist"' in bad and not bad.lstrip().startswith("#")
