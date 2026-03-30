import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def get_project_dir() -> str:
    """
    Where templates/static live.
    - Normal run: folder containing app.py
    - PyInstaller onefile: sys._MEIPASS (temporary bundle dir)
    """
    if is_frozen():
        return sys._MEIPASS  # type: ignore[attr-defined]
    # app.py is in project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_writable_dir() -> str:
    """
    Where writable files live (SQLite DB, ledger, uploads).
    - Normal run: project root
    - PyInstaller exe: folder containing the exe
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(writable_dir: str) -> None:
    os.makedirs(os.path.join(writable_dir, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(writable_dir, "database"), exist_ok=True)

    ledger_path = os.path.join(writable_dir, "database", "ledger.json")
    if not os.path.exists(ledger_path):
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write("[]")