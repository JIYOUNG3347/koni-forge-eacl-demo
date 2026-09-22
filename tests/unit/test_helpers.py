"""Tests for server.core.helpers — format_size, get_dir_size."""

from server.core.helpers import format_size, get_dir_size


class TestFormatSize:
    """Tests for format_size (human-readable byte formatting)."""

    def test_format_size_bytes(self):
        """Small values should show in bytes."""
        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"
        assert format_size(1) == "1 B"

    def test_format_size_kb(self):
        """1024 bytes should show as 1.00 KB."""
        assert format_size(1024) == "1.00 KB"

    def test_format_size_mb(self):
        """1048576 bytes should show as 1.00 MB."""
        assert format_size(1048576) == "1.00 MB"

    def test_format_size_gb(self):
        """Large values should show in GB."""
        assert format_size(1073741824) == "1.00 GB"
        # 2.5 GB
        assert format_size(int(2.5 * 1024**3)) == "2.50 GB"


class TestGetDirSize:
    """Tests for get_dir_size (recursive directory size)."""

    def test_get_dir_size_empty(self, tmp_path):
        """Empty directory should return 0."""
        assert get_dir_size(tmp_path) == 0

    def test_get_dir_size_nonexistent(self, tmp_path):
        """Non-existent path should return 0."""
        assert get_dir_size(tmp_path / "does_not_exist") == 0

    def test_get_dir_size_with_files(self, tmp_path):
        """Directory with files should return total size."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world!")
        size = get_dir_size(tmp_path)
        assert size > 0
        # 5 bytes for "hello" + 6 bytes for "world!"
        assert size == 11
