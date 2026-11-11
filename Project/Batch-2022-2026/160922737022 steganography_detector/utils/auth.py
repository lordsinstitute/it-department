from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from models.models import User

def hash_password(pw):
    return generate_password_hash(pw)

def authenticate(username, password):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        session["user"] = username
        return True
    return False

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper