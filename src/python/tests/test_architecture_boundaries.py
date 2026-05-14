import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BANNED_IMPORT_ROOTS = {"c4az", "connect4_zero"}
BANNED_TEXT = {
    "archive/legacy_python",
    "archive/python_c4az_2026_05_14",
    "archive/rust_prototype_2026_05_14",
}


def active_files():
    for base in [ROOT / "src" / "python", ROOT / "src" / "cpp"]:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".hpp"}:
                if path.name == "test_architecture_boundaries.py":
                    continue
                yield path


def test_active_python_does_not_import_archived_packages():
    for path in (ROOT / "src" / "python").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_IMPORT_ROOTS, path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_IMPORT_ROOTS, path


def test_active_code_does_not_reference_archived_paths():
    for path in active_files():
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_TEXT:
            assert banned not in text, path
