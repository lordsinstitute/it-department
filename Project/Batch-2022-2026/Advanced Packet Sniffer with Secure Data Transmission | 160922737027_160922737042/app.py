import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from uuid import uuid4

from flask import Flask, render_template, redirect, url_for, request, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from models import db
from models.db_models import User, Run
from detector.analyzer import analyze_payload
from utils.ledger import Ledger
from utils.pdf_report import build_pdf_report
from utils.scoring import risk_label_from_score
from utils.paths import get_project_dir, get_writable_dir, ensure_dirs
from utils.errors import safe_route, safe_read_text


# ---------------------------
# App Factory
# ---------------------------
def create_app():
    project_dir = get_project_dir()      # templates/static source
    writable_dir = get_writable_dir()    # db/uploads/ledger target

    ensure_dirs(writable_dir)

    app = Flask(
        __name__,
        template_folder=os.path.join(project_dir, "templates"),
        static_folder=os.path.join(project_dir, "static"),
    )

    # NOTE: demo-friendly secret; for real use set env SECRET_KEY
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")

    # SQLite must be stored in app root as data.db
    db_path = os.path.join(writable_dir, "data.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path.replace("\\", "/")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Uploads
    app.config["UPLOAD_FOLDER"] = os.path.join(writable_dir, "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(writable_dir, "database", "app.log"),
                encoding="utf-8",
            ),
        ],
    )
    app.logger.setLevel(logging.INFO)

    # Init DB
    db.init_app(app)

    # Login manager
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        try:
            return db.session.get(User, int(user_id))
        except Exception:
            return None

    # Evidence ledger (blockchain-inspired)
    ledger = Ledger(os.path.join(writable_dir, "database", "ledger.json"), app.logger)

    # Auto-create DB tables
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            app.logger.exception("DB create_all failed: %s", e)

    # ---------------------------
    # Error pages
    # ---------------------------
    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template(
            "error.html",
            code=500,
            message="Unexpected error occurred. The app recovered safely.",
        ), 500

    # ---------------------------
    # Routes
    # ---------------------------
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    @safe_route
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            confirm = request.form.get("confirm") or ""

            if len(username) < 3:
                flash("Username must be at least 3 characters.", "warning")
                return render_template("register.html")

            if len(password) < 8:
                flash("Password must be at least 8 characters.", "warning")
                return render_template("register.html")

            if password != confirm:
                flash("Passwords do not match.", "warning")
                return render_template("register.html")

            existing = User.query.filter_by(username=username).first()
            if existing:
                flash("Username already exists. Choose another.", "warning")
                return render_template("register.html")

            is_first_user = (User.query.count() == 0)
            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                is_admin=bool(is_first_user),
                created_at=datetime.utcnow(),
            )
            db.session.add(user)
            db.session.commit()

            flash("Account created. Please login.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    @safe_route
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""

            user = User.query.filter_by(username=username).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid username or password.", "danger")
                return render_template("login.html")

            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    @safe_route
    def logout():
        logout_user()
        flash("Logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    @safe_route
    def dashboard():
        total_runs = Run.query.count()
        last_run = Run.query.order_by(Run.created_at.desc()).first()
        last_run_time = last_run.created_at if last_run else None

        high_plus = Run.query.filter(Run.risk.in_(["High", "Critical"])).count()
        critical = Run.query.filter_by(risk="Critical").count()

        risks = ["Low", "Medium", "High", "Critical"]
        risk_counts = [Run.query.filter_by(risk=r).count() for r in risks]

        today = datetime.utcnow().date()
        labels = []
        counts = []
        for i in range(13, -1, -1):
            d = today - timedelta(days=i)
            labels.append(d.strftime("%b %d"))
            start = datetime(d.year, d.month, d.day)
            end = start + timedelta(days=1)
            c = Run.query.filter(Run.created_at >= start, Run.created_at < end).count()
            counts.append(c)

        return render_template(
            "dashboard.html",
            total_runs=total_runs,
            high_plus=high_plus,
            critical=critical,
            last_run_time=last_run_time,
            risks=risks,
            risk_counts=risk_counts,
            run_labels=labels,
            run_counts=counts,
        )

    @app.route("/analyze", methods=["GET", "POST"])
    @login_required
    @safe_route
    def analyze():
        if request.method == "POST":
            input_mode = request.form.get("input_mode", "text")

            raw_text = ""
            file_name = None
            saved_path = None

            if input_mode == "file":
                f = request.files.get("upload")
                if not f or f.filename.strip() == "":
                    flash("No file selected.", "warning")
                    return render_template("analyze.html")

                file_name = secure_filename(f.filename)
                unique_name = f"{uuid4().hex}_{file_name}"
                saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

                try:
                    f.save(saved_path)
                    raw_text = safe_read_text(saved_path)
                except Exception as e:
                    app.logger.exception("File upload/read failed: %s", e)
                    flash("Upload failed or file could not be read as text.", "danger")
                    return render_template("analyze.html")

            else:
                raw_text = (request.form.get("payload") or "").strip()
                if not raw_text:
                    flash("Paste some text to analyze, or switch to file upload.", "warning")
                    return render_template("analyze.html")

            result = analyze_payload(raw_text)
            risk = risk_label_from_score(result["score"])

            # Alert simulation: terminal + UI
            if risk in ("High", "Critical"):
                msg = f"[ALERT SIMULATION] Risk={risk} | Score={result['score']} | Findings={len(result['findings'])}"
                print(msg)
                app.logger.warning(msg)
                flash(f"Alert Simulation: {msg}", "danger")
            else:
                flash(f"Analysis complete. Risk={risk} Score={result['score']}", "success")

            entry_payload = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "username": current_user.username,
                "input_mode": input_mode,
                "source_file": file_name,
                "summary": result["summary"],
                "score": result["score"],
                "risk": risk,
                "findings": result["findings"],
                "indicators": result["indicators"],
            }
            ledger_info = ledger.append(entry_payload)

            run = Run(
                user_id=current_user.id,
                created_at=datetime.utcnow(),
                input_mode=input_mode,
                source_file=file_name,
                stored_file=saved_path,
                score=result["score"],
                risk=risk,
                summary_json=json.dumps(result["summary"], ensure_ascii=False),
                findings_json=json.dumps(result["findings"], ensure_ascii=False),
                indicators_json=json.dumps(result["indicators"], ensure_ascii=False),
                evidence_hash=ledger_info["hash"],
                prev_hash=ledger_info["prev_hash"],
                chain_index=ledger_info["index"],
            )
            db.session.add(run)
            db.session.commit()

            return redirect(url_for("run_detail", run_id=run.id))

        return render_template("analyze.html")

    @app.route("/history")
    @login_required
    @safe_route
    def history():
        page = request.args.get("page", 1, type=int)
        per_page = 10
        pagination = Run.query.order_by(Run.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        return render_template("history.html", pagination=pagination, runs=pagination.items)

    @app.route("/run/<int:run_id>")
    @login_required
    @safe_route
    def run_detail(run_id: int):
        run = db.session.get(Run, run_id)
        if not run:
            flash("Run not found.", "warning")
            return redirect(url_for("history"))

        summary = json.loads(run.summary_json or "{}")
        findings = json.loads(run.findings_json or "[]")
        indicators = json.loads(run.indicators_json or "{}")

        return render_template(
            "run_detail.html",
            run=run,
            summary=summary,
            findings=findings,
            indicators=indicators,
        )

    @app.route("/run/<int:run_id>/report")
    @login_required
    @safe_route
    def download_report(run_id: int):
        run = db.session.get(Run, run_id)
        if not run:
            flash("Run not found.", "warning")
            return redirect(url_for("history"))

        summary = json.loads(run.summary_json or "{}")
        findings = json.loads(run.findings_json or "[]")
        indicators = json.loads(run.indicators_json or "{}")

        pdf_bytes = build_pdf_report(run, summary, findings, indicators)
        filename = f"packet_sniffer_report_run_{run.id}.pdf"

        return send_file(
            pdf_bytes,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )

    return app


# Create the app once (IMPORTANT: do not create twice)
app = create_app()


def _open_browser_later(url: str, delay_s: float = 1.2):
    time.sleep(delay_s)
    try:
        webbrowser.open_new(url)
    except Exception as e:
        print("[BROWSER OPEN ERROR]", e)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}/"

    print("\nStarting PacketFlow Guard...")
    print(f"Opening browser at {url}\n")

    threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()
    app.run(host=host, port=port, debug=False)