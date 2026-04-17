import sys
from pathlib import Path


def get_app_base_dir() -> Path:
    """
    Returns a writable base directory.

    - Normal python run: project folder (where this file's parent is)
    - Frozen EXE: directory containing the executable (writable, stable)
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return Path(sys.executable).resolve().parent
    # project root = parent of this file's folder (utils/)
    return Path(__file__).resolve().parents[1]


def ensure_runtime_dirs(base_dir: Path) -> None:
    (base_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (base_dir / "database").mkdir(parents=True, exist_ok=True)