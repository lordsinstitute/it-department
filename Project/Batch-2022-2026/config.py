import os
import secrets
import sys

def app_root() -> str:
    """
    App root should be writable even in EXE mode.
    - In PyInstaller onefile: sys.executable points to the EXE path.
    - In normal python: this file lives in project root.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))

class Config:
    ROOT_DIR = app_root()

    # Required: DB must be stored in app root as data.db (for EXE stability)
    DB_PATH = os.path.join(ROOT_DIR, "data.db")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + DB_PATH.replace("\\", "/")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(16)

    UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
    DATABASE_FOLDER = os.path.join(ROOT_DIR, "database")
    LEDGER_PATH = os.path.join(DATABASE_FOLDER, "ledger.json")

    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB upload limit