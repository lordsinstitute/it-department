from flask import Blueprint, render_template, request, redirect, url_for, flash
from detector.models import User
from werkzeug.security import check_password_hash, generate_password_hash
from detector import db
from flask_login import login_user, logout_user, login_required

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""

            user = User.query.filter_by(username=username).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid username or password.", "danger")
                return render_template("login.html")

            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("main.dashboard"))

        return render_template("login.html")
    except Exception:
        flash("Login failed due to a system issue. Please try again.", "danger")
        return render_template("login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    Optional: simple registration for demo. Admin already exists.
    """
    try:
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            confirm = request.form.get("confirm") or ""

            if len(username) < 3:
                flash("Username must be at least 3 characters.", "warning")
                return render_template("login.html", register=True)
            if len(password) < 8:
                flash("Password must be at least 8 characters.", "warning")
                return render_template("login.html", register=True)
            if password != confirm:
                flash("Passwords do not match.", "warning")
                return render_template("login.html", register=True)

            if User.query.filter_by(username=username).first():
                flash("Username already exists.", "warning")
                return render_template("login.html", register=True)

            user = User(
                username=username,
                password_hash=generate_password_hash(password, method="pbkdf2:sha256", salt_length=16),
                role="USER",
            )
            db.session.add(user)
            db.session.commit()
            flash("Account created. You can now login.", "success")
            return redirect(url_for("auth.login"))

        return render_template("login.html", register=True)
    except Exception:
        flash("Registration failed due to a system issue. Please try again.", "danger")
        return render_template("login.html", register=True)