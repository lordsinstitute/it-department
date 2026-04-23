from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash

from .models import User

auth_bp = Blueprint("auth", __name__)


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("auth.login"))
        return view_func(*args, **kwargs)
    return wrapper


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            user = User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password_hash, password):
                session["user_id"] = user.id
                session["username"] = user.username
                session["full_name"] = user.full_name
                flash("Login successful.", "success")
                return redirect(url_for("main.dashboard"))

            flash("Invalid username or password.", "danger")

        return render_template("login.html")
    except Exception:
        flash("Unable to process login.", "danger")
        return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.login"))