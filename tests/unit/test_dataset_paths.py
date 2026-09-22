from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from server.core import dataset_paths as dp

_ROOT = Path(__file__).resolve().parents[2]

# ── Root resolution ─────────────────────────────────────────────────────────


def test_storage_root_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    """Never hardcoded. A deployment container provides `/storage`."""
    monkeypatch.setenv("STORAGE_BASE_PATH", "/tmp/koni-test-storage")
    assert dp.storage_root() == Path("/tmp/koni-test-storage")
    monkeypatch.delenv("STORAGE_BASE_PATH", raising=False)
    assert dp.storage_root() == Path(dp.DEFAULT_STORAGE_ROOT)




def test_root_order_argument_is_gone():
    """`generated_first` was a temporary argument for the two-root priority."""
    import inspect

    for fn in (dp.dataset_roots, dp.iter_dataset_dirs, dp.find_dataset_dir):
        assert "generated_first" not in inspect.signature(fn).parameters, fn.__name__


def test_raw_root_is_separate():
    """Source documents differ in kind from datasets and are kept separate."""
    assert dp.raw_root().name == "raw_corpus"
    assert dp.raw_root() not in dp.dataset_roots()


# ── Finding a folder ────────────────────────────────────────────────────────


@pytest.fixture()
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("STORAGE_BASE_PATH", str(tmp_path))
    (tmp_path / "corpus").mkdir()
    (tmp_path / "generated_corpus").mkdir()
    return tmp_path


def test_find_returns_none_instead_of_inventing_a_path(storage: Path):
    """**Nothing is invented** — a miss does not return the first candidate.

    Returning a nonexistent path makes the caller believe it found one and
    report an empty result as normal.
    """
    assert dp.find_dataset_dir("nope") is None
    assert dp.find_dataset_file("nope") is None


def test_find_rejects_traversal(storage: Path):
    for bad in ("../etc", "a/b", "..", ".", ""):
        assert dp.find_dataset_dir(bad) is None


def test_only_the_single_dataset_root_is_searched(storage: Path):
    """One root only: the same name under the legacy root is not found."""
    (storage / "corpus" / "X").mkdir()
    (storage / "generated_corpus" / "legacy").mkdir()
    assert dp.find_dataset_dir("X") == storage / "corpus" / "X"
    assert dp.find_dataset_dir("legacy") is None
    # Files are not deleted: cleaning the disk is optional and unrelated to the code.
    assert (storage / "generated_corpus" / "legacy").is_dir()


def test_iter_visits_only_the_single_root(storage: Path):
    (storage / "corpus" / "c1").mkdir()
    (storage / "generated_corpus" / "g1").mkdir()
    assert [d.name for d in dp.iter_dataset_dirs()] == ["c1"]


def test_missing_root_is_skipped_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STORAGE_BASE_PATH", str(tmp_path / "absent"))
    assert list(dp.iter_dataset_dirs()) == []
    assert dp.find_dataset_dir("X") is None


# ── File selection follows the same rule as the trainer ─────────────────────


def test_file_selection_delegates_to_corpus_files(storage: Path):
    """Deciding separately here would split the sample count from the data trained on."""
    folder = storage / "corpus" / "X"
    folder.mkdir()
    (folder / "qa_dataset.json").write_text("[]", encoding="utf-8")
    (folder / "qa_dataset_preprocessed.json").write_text("[]", encoding="utf-8")
    (folder / "preprocessing_stats.txt").write_text("{}", encoding="utf-8")
    selected = dp.find_dataset_file("X")
    assert selected is not None
    assert selected.name == "qa_dataset_preprocessed.json"


def test_prefer_overrides_the_default_rule(storage: Path):
    """Preserves the caller's existing preference; a behaviour-neutral tool."""
    folder = storage / "corpus" / "X"
    folder.mkdir()
    (folder / "qa_dataset.json").write_text("[]", encoding="utf-8")
    (folder / "qa_dataset_preprocessed.json").write_text("[]", encoding="utf-8")
    preferred = dp.find_dataset_file("X", prefer=("qa_dataset.json",))
    assert preferred is not None
    assert preferred.name == "qa_dataset.json"

    # With no preferred name, it falls back to the default rule.
    fallback = dp.find_dataset_file("X", prefer=("absent.json",))
    assert fallback is not None
    assert fallback.name == "qa_dataset_preprocessed.json"


def test_empty_folder_yields_none(storage: Path):
    (storage / "corpus" / "X").mkdir()
    assert dp.find_dataset_file("X") is None


_SCANNED = (
    "modules/agents/specialists",
    "modules/agents/dispatcher.py",
    "server/routers",
    "server/core/agent_datasets.py",
    "server/core/train_dataset_record.py",
    "celery_app/tasks",
    "pipelines/generator_guided.py",
)

#: `server/core/config.py` is the source itself: it reads the root from
#: host.toml and exports it as an environment variable, so `STORAGE_ROOT / "corpus"`
#: there is not a duplicate.
_EXEMPT = {"server/core/config.py", "server/core/dataset_paths.py"}


def _code_lines(path: Path) -> list[str]:
    """Code lines only, with comments and docstrings removed."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and node.end_lineno:
                doc_lines.update(range(node.lineno, node.end_lineno + 1))
    out = []
    for number, line in enumerate(source.splitlines(), start=1):
        if number in doc_lines:
            continue
        out.append(line.split("#", 1)[0])
    return out


def _python_files() -> list[Path]:
    files: list[Path] = []
    for target in _SCANNED:
        path = _ROOT / target
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return [f for f in files if str(f.relative_to(_ROOT)) not in _EXEMPT]


#: Expressions that assemble the root by hand. Both an absolute literal and
#: `STORAGE_ROOT / "corpus"` are blocked; the latter is easier to miss.
_FORBIDDEN = (
    '"/storage/corpus"',
    '"/storage/generated_corpus"',
    'STORAGE_ROOT / "corpus"',
    'STORAGE_ROOT / "generated_corpus"',
)


def test_dataset_paths_is_import_light():
    """It must import in a plain CI venv."""
    tree = ast.parse((_ROOT / "server" / "core" / "dataset_paths.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.col_offset == 0:
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "os", "pathlib", "typing"}, modules


def test_storage_base_follows_host_toml():
    """A pure module cannot read host.toml, so `server.core.config` exports the
    value as an environment variable. In unit tests that value is
    `tests/fixtures/host.toml`, which makes this assertion the proof that the
    two sources are one.
    """
    from server.core.config import STORAGE_ROOT

    assert os.getenv("STORAGE_BASE_PATH") == str(STORAGE_ROOT)
    assert dp.storage_root() == STORAGE_ROOT
