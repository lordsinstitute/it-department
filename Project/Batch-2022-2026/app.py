import webbrowser
import threading
import os
import json
import traceback
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from detector.analyzer import analyze_input
from utils.helpers import (
    resource_path, ensure_directories, allowed_file,
    get_db_path, get_upload_folder, get_ledger_path
)
from utils.ledger import add_record_to_chain
from utils.reporting import generate_pdf_report

BASE_DIR = resource_path(".")
db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "change-this-in-production-demo"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{get_db_path()}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = get_upload_folder()
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

    ensure_directories()
    db.init_app(app)

    return app


app = create_app()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AnalysisHistory(db.Model):
    __tablename__ = "analysis_history"

    id = db.Column(db.Integer, primary_key=True)
    source_type = db.Column(db.String(20), nullable=False)  # text/file
    input_name = db.Column(db.String(255), nullable=True)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)
    packets_analyzed = db.Column(db.Integer, default=0)
    findings_json = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=False)
    ledger_hash = db.Column(db.String(128), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "error.html",
        message="Uploaded file is too large. Please upload a file under 20 MB."
    ), 413


@app.errorhandler(Exception)
def handle_global_exception(error):
    app.logger.error("Unhandled exception: %s\n%s", str(error), traceback.format_exc())
    return render_template(
        "error.html",
        message="Something went wrong, but the application is still running safely."
    ), 500


def initialize_database():
    with app.app_context():
        db.create_all()

        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                password_hash=generate_password_hash("admin123")
            )
            db.session.add(admin)
            db.session.commit()

        ledger_path = get_ledger_path()
        if not os.path.exists(ledger_path):
            os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                session["user_id"] = user.id
                session["username"] = user.username
                flash("Login successful.", "success")
                return redirect(url_for("dashboard"))

            flash("Invalid username or password.", "danger")
        except Exception:
            flash("Login failed due to an internal error.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    try:
        all_runs = AnalysisHistory.query.order_by(AnalysisHistory.created_at.desc()).all()
        total_runs = len(all_runs)
        total_packets = sum(run.packets_analyzed for run in all_runs)
        critical_count = sum(1 for run in all_runs if run.risk_level == "Critical")
        high_count = sum(1 for run in all_runs if run.risk_level == "High")

        risk_distribution = {
            "Low": 0,
            "Medium": 0,
            "High": 0,
            "Critical": 0
        }
        source_distribution = {
            "text": 0,
            "file": 0
        }

        for run in all_runs:
            risk_distribution[run.risk_level] = risk_distribution.get(run.risk_level, 0) + 1
            source_distribution[run.source_type] = source_distribution.get(run.source_type, 0) + 1

        recent_runs = all_runs[:5]

        return render_template(
            "dashboard.html",
            total_runs=total_runs,
            total_packets=total_packets,
            critical_count=critical_count,
            high_count=high_count,
            risk_distribution=risk_distribution,
            source_distribution=source_distribution,
            recent_runs=recent_runs
        )
    except Exception:
        flash("Dashboard data could not be loaded.", "danger")
        return render_template(
            "dashboard.html",
            total_runs=0,
            total_packets=0,
            critical_count=0,
            high_count=0,
            risk_distribution={"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            source_distribution={"text": 0, "file": 0},
            recent_runs=[]
        )


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        try:
            text_input = request.form.get("text_input", "").strip()
            uploaded_file = request.files.get("file_input")

            file_path = None
            input_name = None
            source_type = None

            if uploaded_file and uploaded_file.filename:
                if not allowed_file(uploaded_file.filename):
                    flash("Unsupported file type. Use .pcap, .pcapng, .txt, .log, .csv, or .json", "danger")
                    return redirect(url_for("analyze"))

                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                uploaded_file.save(file_path)
                input_name = filename
                source_type = "file"

            elif text_input:
                input_name = "manual_text_input"
                source_type = "text"

            else:
                flash("Please provide either text input or upload a file.", "warning")
                return redirect(url_for("analyze"))

            analysis_result = analyze_input(
                text_input=text_input,
                file_path=file_path
            )

            chain_record = {
                "input_name": input_name,
                "source_type": source_type,
                "summary": analysis_result["summary"],
                "risk_score": analysis_result["risk_score"],
                "risk_level": analysis_result["risk_level"],
                "packets_analyzed": analysis_result["packets_analyzed"],
                "findings": analysis_result["findings"],
                "timestamp": datetime.utcnow().isoformat()
            }

            ledger_hash = add_record_to_chain(chain_record)

            history = AnalysisHistory(
                source_type=source_type,
                input_name=input_name,
                risk_level=analysis_result["risk_level"],
                risk_score=analysis_result["risk_score"],
                packets_analyzed=analysis_result["packets_analyzed"],
                findings_json=json.dumps(analysis_result["findings"], indent=2),
                summary=analysis_result["summary"],
                ledger_hash=ledger_hash
            )
            db.session.add(history)
            db.session.commit()

            flash("Analysis completed successfully.", "success")
            return redirect(url_for("result", history_id=history.id))

        except Exception as exc:
            app.logger.error("Analysis failure: %s\n%s", str(exc), traceback.format_exc())
            flash("Analysis failed safely. Please try a different file or input.", "danger")
            return redirect(url_for("analyze"))

    return render_template("analyze.html")


@app.route("/result/<int:history_id>")
@login_required
def result(history_id):
    try:
        item = AnalysisHistory.query.get_or_404(history_id)
        findings = json.loads(item.findings_json)
        return render_template("result.html", item=item, findings=findings)
    except Exception:
        flash("Could not load analysis result.", "danger")
        return redirect(url_for("history"))


@app.route("/history")
@login_required
def history():
    try:
        items = AnalysisHistory.query.order_by(AnalysisHistory.created_at.desc()).all()
        return render_template("history.html", items=items)
    except Exception:
        flash("Could not load history.", "danger")
        return render_template("history.html", items=[])


@app.route("/download-report/<int:history_id>")
@login_required
def download_report(history_id):
    try:
        item = AnalysisHistory.query.get_or_404(history_id)
        findings = json.loads(item.findings_json)

        output_path = generate_pdf_report(
            report_id=item.id,
            input_name=item.input_name or "unknown",
            source_type=item.source_type,
            summary=item.summary,
            risk_level=item.risk_level,
            risk_score=item.risk_score,
            packets_analyzed=item.packets_analyzed,
            findings=findings,
            ledger_hash=item.ledger_hash,
            created_at=item.created_at
        )

        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"packet_analysis_report_{item.id}.pdf"
        )
    except Exception:
        flash("Could not generate PDF report.", "danger")
        return redirect(url_for("history"))

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    initialize_database()

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1, open_browser).start()

    app.run(debug=True)