import sys
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]

def ensure_app_folders(base_dir: Path) -> None:
    (base_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (base_dir / "database").mkdir(parents=True, exist_ok=True)
    (base_dir / "static").mkdir(parents=True, exist_ok=True)
    (base_dir / "templates").mkdir(parents=True, exist_ok=True)
    (base_dir / "models").mkdir(parents=True, exist_ok=True)
    (base_dir / "detector").mkdir(parents=True, exist_ok=True)
    (base_dir / "utils").mkdir(parents=True, exist_ok=True)

def get_db_uri(base_dir: Path) -> str:
    db_path = base_dir / "data.db"
    return f"sqlite:///{db_path}"

def safe_join(base: Path, filename: str) -> Path:
    p = (base / filename).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("Unsafe path.")
    return p

def is_allowed_extension(ext_with_dot: str, allowed: set) -> bool:
    ext = ext_with_dot.lower().lstrip(".").strip()
    return ext in allowed