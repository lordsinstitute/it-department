import traceback
from functools import wraps
from flask import flash, render_template
from flask_login import current_user


def safe_route(fn):
    """
    Prevent demo crashes: show a friendly error page instead of looping redirects.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            flash("The app recovered from an error safely. See details below.", "danger")
            return render_template(
                "error.html",
                code=500,
                message="A route failed unexpectedly (template missing or runtime error). Fix the file paths or templates and retry.",
            ), 500
    return wrapper


def safe_read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return raw.decode("latin-1", errors="ignore")
        except Exception:
            return ""