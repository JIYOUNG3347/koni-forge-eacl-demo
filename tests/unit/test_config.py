"""Tests for server.core.config — path constants and env defaults."""

from pathlib import Path

from server.core import config


class TestConfig:
    """Tests for configuration constants."""

    def test_storage_root_exists(self):
        """STORAGE_ROOT must be a Path object."""
        assert isinstance(config.STORAGE_ROOT, Path)

    def test_all_dirs_are_paths(self):
        """All directory constants must be Path objects."""
        path_attrs = [
            "STORAGE_ROOT",
            "MODELS_DIR",
            "CORPUS_DIR",
            "DATASETS_DIR",
            "CHECKPOINTS_DIR",
            "OUTPUTS_DIR",
            "CHROMA_DIR",
            "LOGS_DIR",
            "TEMP_DIR",
            "USERS_DIR",
        ]
        for attr_name in path_attrs:
            val = getattr(config, attr_name)
            assert isinstance(val, Path), f"{attr_name} should be Path, got {type(val)}"

    def test_env_defaults(self):
        """Key configuration values must have sensible defaults."""
        # INTERNAL_API_URL should be a string containing http
        assert isinstance(config.INTERNAL_API_URL, str)
        assert config.INTERNAL_API_URL.startswith("http")

        # REDIS_URL should be a string
        assert isinstance(config.REDIS_URL, str)

        # VERSION should be a non-empty string
        assert isinstance(config.VERSION, str)
        assert len(config.VERSION) > 0

        # BCRYPT_COST_FACTOR should be a positive int
        assert isinstance(config.BCRYPT_COST_FACTOR, int)
        assert config.BCRYPT_COST_FACTOR > 0
