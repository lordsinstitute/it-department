from flask import Flask, render_template, redirect, url_for, request, send_file
from flask_login import LoginManager, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from models.models import db, User, AnalysisHistory
from detector.analyzer import analyze_data
from utils.blockchain import add_to_ledger
from utils.pdf_report import generate_pdf
from utils.alerts import trigger_alert
from config import Config
import os

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        hashed = generate_password_hash(request.form["password"])
        db.session.add(User(username=request.form["username"], password=hashed))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/dashboard")
@login_required
def dashboard():
    total = AnalysisHistory.query.count()
    return render_template("dashboard.html", total=total)

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        text = request.form.get("text", "")
        risk, findings = analyze_data(text)
        block_hash = add_to_ledger(text)

        if risk in ["High", "Critical"]:
            trigger_alert("High-risk data detected")

        record = AnalysisHistory(
            input_type="Text",
            content=text,
            risk_level=risk,
            hash_value=block_hash
        )
        db.session.add(record)
        db.session.commit()

        return render_template("result.html", risk=risk, findings=findings, hash=block_hash)

    return render_template("analyze.html")

@app.route("/history")
@login_required
def history():
    records = AnalysisHistory.query.all()
    return render_template("history.html", records=records)

@app.route("/report/<int:id>")
@login_required
def report(id):
    record = AnalysisHistory.query.get_or_404(id)
    filename = f"report_{id}.pdf"
    generate_pdf(filename, record.__dict__)
    return send_file(filename, as_attachment=True)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))

import threading
import webbrowser
import time

def open_browser():
    time.sleep(1.5)  # wait for Flask server to start
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)