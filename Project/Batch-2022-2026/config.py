import os
from utils.paths import get_writable_app_root

class Config:
    # Writable root (important for EXE + SQLite)
    APP_ROOT = get_writable_app_root()

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me-please-32chars-min")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Must be stored in app root as data.db (per your constraint)
    DB_PATH = os.path.join(APP_ROOT, "data.db")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + DB_PATH.replace("\\", "/")

    UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
    LEDGER_PATH = os.path.join(APP_ROOT, "ledger.json")

    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB