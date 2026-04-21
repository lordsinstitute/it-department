from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from detector.extensions import db
from models.user import User

auth_bp = Blueprint("auth", __name__)

@auth_bp.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template("login.html")

@auth_bp.post("/login")
def login_post():
    try:
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("auth.login"))

        login_user(user)
        flash("Welcome back.", "success")
        return redirect(url_for("main.dashboard"))
    except Exception:
        flash("Login failed due to an unexpected error.", "danger")
        return redirect(url_for("auth.login"))

@auth_bp.get("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))

@auth_bp.get("/settings")
@login_required
def settings():
    return render_template("settings.html")

@auth_bp.post("/settings/password")
@login_required
def change_password():
    try:
        current = request.form.get("current_password") or ""
        newp = request.form.get("new_password") or ""
        confirm = request.form.get("confirm_password") or ""

        if not current_user.check_password(current):
            flash("Current password is incorrect.", "danger")
            return redirect(url_for("auth.settings"))

        if len(newp) < 8:
            flash("New password must be at least 8 characters.", "warning")
            return redirect(url_for("auth.settings"))

        if newp != confirm:
            flash("New password and confirmation do not match.", "warning")
            return redirect(url_for("auth.settings"))

        current_user.set_password(newp)
        db.session.commit()
        flash("Password updated successfully.", "success")
        return redirect(url_for("auth.settings"))
    except Exception:
        flash("Could not update password due to an error.", "danger")
        return redirect(url_for("auth.settings"))