from pathlib import Path
import os
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return Path(__file__).resolve().parent.parent


def get_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(os.path.dirname(sys.executable))
    return Path(__file__).resolve().parent.parent


class Config:
    BASE_DIR = get_base_dir()          # for bundled resources like templates/static
    RUNTIME_ROOT = get_runtime_root()  # for writable files like data.db

    SECRET_KEY = "email-shield-secure-demo-key-change-in-production"

    DATABASE_DIR = RUNTIME_ROOT / "database"
    MODELS_DIR = BASE_DIR / "models"
    UPLOADS_DIR = RUNTIME_ROOT / "uploads"
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{(RUNTIME_ROOT / 'data.db').as_posix()}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024


def ensure_directories():
    for folder_path in [
        Config.DATABASE_DIR,
        Config.UPLOADS_DIR,
        Config.RUNTIME_ROOT
    ]:
        folder_path.mkdir(parents=True, exist_ok=True)