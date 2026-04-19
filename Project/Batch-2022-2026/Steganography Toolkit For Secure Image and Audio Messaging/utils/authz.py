from functools import wraps
from flask import abort, flash
from flask_login import current_user

def role_required(*roles):
    roles_set = {r.lower() for r in roles}

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(403)
            user_role = (getattr(current_user, "role", "Analyst") or "Analyst").lower()
            if user_role not in roles_set:
                flash("Access denied: insufficient role.", "danger")
                return abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator