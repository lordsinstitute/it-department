import os
import sys

def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

def get_writable_app_root() -> str:
    """
    IMPORTANT for EXE:
    - In PyInstaller onefile/onedir, bundled files live in _MEIPASS (read-only-ish)
    - Writable location should be current working directory (user runs the exe from a folder)
    Requirement: database must be stored in app root as data.db -> we treat CWD as app root in frozen mode.
    """
    if is_frozen():
        return os.getcwd()
    # In normal python run, project root = folder containing app.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resource_path(relative_path: str) -> str:
    """
    For templates/static when frozen.
    """
    if is_frozen():
        base = sys._MEIPASS  # type: ignore[attr-defined]
        return os.path.join(base, relative_path)
    return os.path.join(get_writable_app_root(), relative_path)