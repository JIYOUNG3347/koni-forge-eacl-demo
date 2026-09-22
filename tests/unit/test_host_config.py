"""host.toml parser."""

from pathlib import Path

import pytest

from server.core.host_config import HostConfig, find_and_load, load

VALID_TOML = """\
[model_limits]
max_train_param_b = 14
max_train_seq_length = 2048
max_train_batch_size = 8

[concurrency]
max_concurrent_train_jobs = 1
max_concurrent_kbd_jobs = 1

[paths]
storage_base = "/storage"
"""


class TestHostConfigLoad:
    def test_valid_toml_returns_host_config(self, tmp_path: Path) -> None:
        p = tmp_path / "host.toml"
        p.write_text(VALID_TOML)
        cfg = load(p)
        assert isinstance(cfg, HostConfig)
        assert cfg.model_limits.max_train_param_b == 14
        assert cfg.model_limits.max_train_seq_length == 2048
        assert cfg.concurrency.max_concurrent_train_jobs == 1
        assert cfg.paths.storage_base == "/storage"

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        """A path given explicitly must exist — silently defaulting would hide a typo."""
        with pytest.raises(SystemExit) as exc_info:
            load(tmp_path / "nonexistent.toml")
        assert exc_info.value.code == 1

    def test_invalid_toml_syntax_exits(self, tmp_path: Path) -> None:
        p = tmp_path / "host.toml"
        p.write_text("this is not valid toml ][[[")
        with pytest.raises(SystemExit) as exc_info:
            load(p)
        assert exc_info.value.code == 1

    def test_partial_toml_fills_defaults(self, tmp_path: Path) -> None:
        """Every section is optional; the rest falls back to the defaults."""
        p = tmp_path / "host.toml"
        p.write_text("[model_limits]\nmax_train_batch_size = 2\n")
        cfg = load(p)
        assert cfg.model_limits.max_train_batch_size == 2
        assert cfg.model_limits.max_train_seq_length == 2048
        assert cfg.concurrency.max_concurrent_train_jobs == 1

    def test_invalid_value_exits(self, tmp_path: Path) -> None:
        """A present but nonsensical value stops the boot rather than defaulting."""
        p = tmp_path / "host.toml"
        p.write_text(VALID_TOML.replace("max_train_batch_size = 8", "max_train_batch_size = 0"))
        with pytest.raises(SystemExit) as exc_info:
            load(p)
        assert exc_info.value.code == 1


class TestHostConfigDiscovery:
    def test_override_env_wins(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "custom.toml"
        p.write_text(VALID_TOML.replace("max_train_param_b = 14", "max_train_param_b = 3"))
        monkeypatch.setenv("KONI_HOST_CONFIG_PATH", str(p))
        assert find_and_load().model_limits.max_train_param_b == 3

    def test_no_file_returns_defaults(self, monkeypatch) -> None:
        """A fresh clone with no host.toml must still boot."""
        monkeypatch.delenv("KONI_HOST_CONFIG_PATH", raising=False)
        monkeypatch.setattr("server.core.host_config.PROJECT_ROOT", Path("/nonexistent"))
        monkeypatch.setattr(Path, "exists", lambda self: False)
        cfg = find_and_load()
        assert cfg.model_limits.max_train_param_b == 14
        assert cfg.concurrency.max_concurrent_kbd_jobs == 1
