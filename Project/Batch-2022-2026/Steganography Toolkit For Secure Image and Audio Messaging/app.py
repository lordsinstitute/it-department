import os
import json
import traceback
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models.db import db
from models.user import User
from models.history import RunHistory

from utils.paths import get_base_dir, ensure_app_folders, get_db_uri, safe_join, is_allowed_extension
from utils.helpers import secure_save_upload, now_utc_iso, json_dumps_safe, safe_stat
from utils.ledger import init_ledger_if_missing, add_to_ledger, read_ledger_tail
from utils.pdf_report import build_pdf_report
from utils.risk import compute_risk_label
from utils.authz import role_required
from utils.artifacts import build_artifact_zip

from detector.encoder import (
    encode_message_batch_to_carrier,
    estimate_required_bits,
    advise_min_image_dims,
    advise_min_wav_duration
)

import hashlib

APP_NAME = "Stego Toolkit"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_role_column_sqlite(app: Flask) -> None:
    try:
        with app.app_context():
            engine = db.engine
            with engine.connect() as conn:
                res = conn.exec_driver_sql("PRAGMA table_info(users);").fetchall()
                cols = [r[1] for r in res]
                if "role" not in cols:
                    conn.exec_driver_sql("ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'Analyst';")
    except Exception:
        traceback.print_exc()

def alert_simulation(risk_level: str, message: str) -> None:
    print(f"[ALERT-SIM] [{risk_level}] {message}")

def _parse_records_from_form(form) -> list:
    labels = form.getlist("label[]")
    datas = form.getlist("data[]")
    records = []
    for i in range(min(len(labels), len(datas))):
        label = (labels[i] or f"message_{i+1}").strip()[:80]
        data = (datas[i] or "").strip()
        if not data:
            continue
        records.append({
            "label": label,
            "content_type": "text/plain; charset=utf-8",
            "data": data
        })
    if not records:
        records = [{"label": "message_1", "content_type": "text/plain; charset=utf-8", "data": ""}]
    return records

def create_app() -> Flask:
    base_dir = get_base_dir()
    ensure_app_folders(base_dir)
    init_ledger_if_missing(base_dir / "database" / "ledger.json")

    app = Flask(__name__)
    app.config["BASE_DIR"] = str(base_dir)
    app.config["SECRET_KEY"] = os.environ.get("STEGO_SECRET_KEY", "dev-change-me-please")
    app.config["SQLALCHEMY_DATABASE_URI"] = get_db_uri(base_dir)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = str(base_dir / "uploads")

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    with app.app_context():
        db.create_all()
    _ensure_role_column_sqlite(app)

    # ---- Errors ----
    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", title="Forbidden", message="You do not have access to this page."), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", title="Not Found", message="The page was not found."), 404

    @app.errorhandler(413)
    def too_large(_e):
        return render_template("error.html", title="File Too Large", message="Upload exceeds limit (25MB)."), 413

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("error.html", title="Server Error", message="Something went wrong, but the app stayed safe."), 500

    # ---- Auth ----
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        try:
            if request.method == "POST":
                username = (request.form.get("username") or "").strip()
                password = request.form.get("password") or ""
                confirm = request.form.get("confirm") or ""

                if not username or not password:
                    flash("Username and password are required.", "danger")
                    return render_template("register.html", app_name=APP_NAME)
                if len(username) < 3:
                    flash("Username must be at least 3 characters.", "warning")
                    return render_template("register.html", app_name=APP_NAME)
                if password != confirm:
                    flash("Passwords do not match.", "danger")
                    return render_template("register.html", app_name=APP_NAME)
                if len(password) < 8:
                    flash("Password must be at least 8 characters.", "warning")
                    return render_template("register.html", app_name=APP_NAME)
                if User.query.filter_by(username=username).first():
                    flash("Username already exists.", "danger")
                    return render_template("register.html", app_name=APP_NAME)

                is_first = (User.query.count() == 0)
                role = "Admin" if is_first else "Analyst"

                user = User(
                    username=username,
                    password_hash=generate_password_hash(password),
                    role=role,
                    created_at=datetime.utcnow()
                )
                db.session.add(user)
                db.session.commit()
                flash(f"Registered successfully. Role: {role}. Please login.", "success")
                return redirect(url_for("login"))

            return render_template("register.html", app_name=APP_NAME)
        except Exception:
            traceback.print_exc()
            flash("Registration failed safely.", "danger")
            return render_template("register.html", app_name=APP_NAME)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        try:
            if request.method == "POST":
                username = (request.form.get("username") or "").strip()
                password = request.form.get("password") or ""
                user = User.query.filter_by(username=username).first()
                if not user or not check_password_hash(user.password_hash, password):
                    flash("Invalid username or password.", "danger")
                    return render_template("login.html", app_name=APP_NAME)
                login_user(user)
                flash(f"Welcome. Role: {user.role}", "success")
                return redirect(url_for("dashboard"))
            return render_template("login.html", app_name=APP_NAME)
        except Exception:
            traceback.print_exc()
            flash("Login failed safely.", "danger")
            return render_template("login.html", app_name=APP_NAME)

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("Logged out.", "info")
        return redirect(url_for("login"))

    # ---- Dashboard ----
    @app.route("/dashboard")
    @login_required
    def dashboard():
        try:
            total_runs = RunHistory.query.filter_by(user_id=current_user.id).count()
            last_24h = RunHistory.query.filter(
                RunHistory.user_id == current_user.id,
                RunHistory.created_at >= datetime.utcnow() - timedelta(hours=24)
            ).count()

            rows = RunHistory.query.filter_by(user_id=current_user.id).all()
            risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
            media_counts = {"image": 0, "audio": 0}

            days = [(datetime.utcnow() - timedelta(days=i)).date() for i in range(6, -1, -1)]
            trend = {d.isoformat(): 0 for d in days}

            for r in rows:
                risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
                media_counts[r.media_type] = media_counts.get(r.media_type, 0) + 1
                d = r.created_at.date().isoformat()
                if d in trend:
                    trend[d] += 1

            recent = RunHistory.query.filter_by(user_id=current_user.id).order_by(RunHistory.created_at.desc()).limit(5).all()

            return render_template(
                "dashboard.html",
                app_name=APP_NAME,
                total_runs=total_runs,
                last_24h=last_24h,
                chart_risk={"labels": list(risk_counts.keys()), "data": [risk_counts[k] for k in risk_counts]},
                chart_media={"labels": list(media_counts.keys()), "data": [media_counts[k] for k in media_counts]},
                chart_trend={"labels": list(trend.keys()), "data": [trend[k] for k in trend]},
                recent=recent
            )
        except Exception:
            traceback.print_exc()
            flash("Dashboard loaded with safe fallback.", "warning")
            return render_template(
                "dashboard.html",
                app_name=APP_NAME,
                total_runs=0,
                last_24h=0,
                chart_risk={"labels": ["Low","Medium","High","Critical"], "data":[0,0,0,0]},
                chart_media={"labels": ["image","audio"], "data":[0,0]},
                chart_trend={"labels": [], "data": []},
                recent=[]
            )

    # ---- Encode (with Advisor) ----
    @app.route("/encode", methods=["GET", "POST"])
    @login_required
    def encode():
        if request.method == "GET":
            return render_template("encode.html", app_name=APP_NAME, advice=None)

        try:
            action = (request.form.get("action") or "encode").strip().lower()

            records = _parse_records_from_form(request.form)
            password = request.form.get("password") or ""
            bits_per_unit = int(request.form.get("bits_per_unit") or "1")
            redundancy_r = int(request.form.get("redundancy_r") or "1")
            allow_compress = bool(request.form.get("allow_compress") == "on")
            adaptive_embedding = bool(request.form.get("adaptive_embedding") == "on")

            # ADVISOR: compute packed size + effective bits + recommend carriers
            if action == "advise":
                # We pack without needing an upload (carrier independent)
                # Use encoder’s pack step by calling encode pipeline partially:
                # easiest: pack by passing a dummy carrier later is complex; instead we estimate from content.
                # Conservative estimate: plaintext container + crypto overhead ~ included by packer itself.
                # We'll re-run packer by calling encode function with a small dummy check is not desired.
                # So we compute using the packer directly by importing it.
                from utils.crypto import pack_payload_batch
                from detector.encoder import advise_min_image_dims, advise_min_wav_duration, estimate_required_bits

                # meta for watermark calculation
                encode_meta = {
                    "media_type": "image",
                    "bits_per_unit": bits_per_unit,
                    "redundancy_r": redundancy_r,
                    "adaptive_embedding": adaptive_embedding,
                    "allow_compress": allow_compress
                }

                pack = pack_payload_batch(
                    records=records,
                    password=password,
                    original_filename="batch.txt",
                    allow_compress=allow_compress,
                    encode_meta=encode_meta
                )

                required_bits = estimate_required_bits(len(pack.packed), redundancy_r)
                img_w, img_h = advise_min_image_dims(required_bits, bits_per_unit)
                wav_sec = advise_min_wav_duration(required_bits, framerate=44100, channels=2)

                advice = {
                    "packed_bytes": int(len(pack.packed)),
                    "required_bits": int(required_bits),
                    "image_min_w": int(img_w),
                    "image_min_h": int(img_h),
                    "wav_min_duration": round(float(wav_sec), 3),
                    "bits_per_unit": int(bits_per_unit),
                    "redundancy_r": int(redundancy_r),
                }

                flash("Recommendation computed. Now choose a carrier file.", "success")
                return render_template("encode.html", app_name=APP_NAME, advice=advice)

            # ENCODE: needs carrier upload
            uploaded = request.files.get("file")
            if not uploaded or uploaded.filename == "":
                flash("Carrier file is required for encoding.", "warning")
                return redirect(url_for("encode"))

            ext = os.path.splitext(uploaded.filename)[1].lower()
            if not is_allowed_extension(ext, {"png", "bmp", "wav"}):
                flash("Carrier must be PNG/BMP/WAV.", "danger")
                return redirect(url_for("encode"))

            saved_path = secure_save_upload(uploaded, app.config["UPLOAD_FOLDER"])

            out_path, findings, pack = encode_message_batch_to_carrier(
                carrier_path=saved_path,
                records=records,
                password=password,
                bits_per_unit=bits_per_unit,
                allow_compress=allow_compress,
                adaptive_embedding=adaptive_embedding,
                redundancy_r=redundancy_r
            )

            # checksums (DB findings)
            checksums = {
                "input_sha256": sha256_file(saved_path),
                "output_sha256": sha256_file(out_path),
                "input_size_bytes": safe_stat(saved_path).get("size_bytes", 0),
                "output_size_bytes": safe_stat(out_path).get("size_bytes", 0),
            }
            findings["checksums"] = checksums

            risk_score = int(findings.get("risk_score", 10))
            risk_level = compute_risk_label(risk_score)

            # Ledger evidence payload (avoid circular dependence on output checksum)
            evidence_payload = {
                "mode": "encode",
                "media_type": findings.get("media_type"),
                "input_file": os.path.basename(saved_path),
                "output_file": os.path.basename(out_path),
                "watermark_id": findings.get("payload", {}).get("watermark_id"),
                "redundancy_r": redundancy_r,
                "adaptive_embedding": bool(adaptive_embedding) if findings.get("media_type") == "image" else False,
                "bits_per_unit": bits_per_unit,
                "input_sha256": checksums.get("input_sha256"),
                "record_count": findings.get("batch", {}).get("record_count"),
                "timestamp": now_utc_iso(),
            }
            ledger_hash, prev_hash = add_to_ledger(base_dir=get_base_dir(), evidence_payload=evidence_payload)

            run = RunHistory(
                user_id=current_user.id,
                run_type="encode",
                media_type=findings.get("media_type", "image"),
                input_text_len=sum(len((r.get("data") or "").encode("utf-8")) for r in records),
                input_filename=os.path.basename(saved_path),
                output_filename=os.path.basename(out_path),
                risk_score=risk_score,
                risk_level=risk_level,
                findings_json=json_dumps_safe(findings),
                ledger_hash=ledger_hash,
                ledger_prev_hash=prev_hash,
                created_at=datetime.utcnow(),
            )
            db.session.add(run)
            db.session.commit()

            alert_simulation(risk_level, f"Encode completed: {run.output_filename}")
            flash("Encoding completed. Output generated successfully.", "success")
            return redirect(url_for("run_detail", run_id=run.id))

        except Exception as e:
            traceback.print_exc()
            flash(f"Encoding failed safely: {type(e).__name__}", "danger")
            return redirect(url_for("encode"))

    # ---- History ----
    @app.route("/history")
    @login_required
    def history():
        try:
            runs = RunHistory.query.filter_by(user_id=current_user.id).order_by(RunHistory.created_at.desc()).all()
            return render_template("history.html", app_name=APP_NAME, runs=runs)
        except Exception:
            traceback.print_exc()
            flash("History loaded with safe fallback.", "warning")
            return render_template("history.html", app_name=APP_NAME, runs=[])

    @app.route("/run/<int:run_id>")
    @login_required
    def run_detail(run_id: int):
        try:
            run = RunHistory.query.filter_by(id=run_id, user_id=current_user.id).first()
            if not run:
                abort(404)

            try:
                findings = json.loads(run.findings_json or "{}")
            except Exception:
                findings = {"note": "Could not parse findings JSON safely."}

            ledger_tail = read_ledger_tail(get_base_dir(), n=6)
            return render_template("run_detail.html", app_name=APP_NAME, run=run, findings=findings, ledger_tail=ledger_tail)
        except Exception:
            traceback.print_exc()
            flash("Run details failed safely.", "danger")
            return redirect(url_for("history"))

    # ---- Downloads ----
    @app.route("/download-file/<path:filename>")
    @login_required
    def download_file(filename: str):
        try:
            base_dir = get_base_dir()
            full = safe_join(base_dir / "uploads", filename)
            if not full.exists():
                abort(404)
            return send_file(str(full), as_attachment=True, download_name=filename)
        except Exception:
            traceback.print_exc()
            flash("Download failed safely.", "danger")
            return redirect(url_for("history"))

    # ---- PDF Report ----
    @app.route("/download/<int:run_id>")
    @login_required
    def download_report(run_id: int):
        try:
            run = RunHistory.query.filter_by(id=run_id, user_id=current_user.id).first()
            if not run:
                abort(404)
            try:
                findings = json.loads(run.findings_json or "{}")
            except Exception:
                findings = {"note": "Findings JSON unavailable."}

            base_dir = get_base_dir()
            out_pdf = base_dir / "uploads" / f"report_run_{run.id}.pdf"
            build_pdf_report(output_path=str(out_pdf), app_name=APP_NAME, run=run, findings=findings)
            return send_file(str(out_pdf), as_attachment=True, download_name=f"stego_report_{run.id}.pdf")
        except Exception:
            traceback.print_exc()
            flash("PDF generation failed safely.", "danger")
            return redirect(url_for("run_detail", run_id=run_id))

    # ---- Export Artifact ZIP ----
    @app.route("/export/<int:run_id>")
    @login_required
    def export_artifact(run_id: int):
        try:
            run = RunHistory.query.filter_by(id=run_id, user_id=current_user.id).first()
            if not run:
                abort(404)
            base_dir = get_base_dir()
            uploads = base_dir / "uploads"

            try:
                findings = json.loads(run.findings_json or "{}")
            except Exception:
                findings = {"note": "Findings JSON unavailable."}

            input_path = uploads / run.input_filename
            output_path = uploads / run.output_filename
            zip_path = uploads / f"artifact_run_{run.id}.zip"

            build_artifact_zip(zip_path, input_path, output_path, findings, run.ledger_hash)
            flash("Artifact exported successfully (ZIP).", "success")
            return send_file(str(zip_path), as_attachment=True, download_name=f"stego_artifact_run_{run.id}.zip")

        except Exception:
            traceback.print_exc()
            flash("Artifact export failed safely.", "danger")
            return redirect(url_for("run_detail", run_id=run_id))

    # ---- Verify ----
    @app.route("/verify", methods=["GET", "POST"])
    @login_required
    def verify():
        upload_result = None
        run_result = None

        runs = RunHistory.query.filter_by(user_id=current_user.id).order_by(RunHistory.created_at.desc()).limit(50).all()
        if request.method == "GET":
            return render_template("verify.html", app_name=APP_NAME, upload_result=upload_result, run_result=run_result, runs=runs)

        try:
            mode = (request.form.get("mode") or "").strip().lower()
            base_dir = get_base_dir()
            uploads = base_dir / "uploads"

            if mode == "upload":
                f = request.files.get("file")
                expected = (request.form.get("expected") or "").strip().lower().replace(" ", "")
                if not f or f.filename == "":
                    flash("Choose a file to verify.", "warning")
                    return redirect(url_for("verify"))

                p = secure_save_upload(f, str(uploads))
                sha = sha256_file(p)
                size_bytes = safe_stat(p).get("size_bytes", 0)
                upload_result = {"sha256": sha, "size_bytes": size_bytes, "expected": expected or None, "match": (sha == expected) if expected else None}
                flash("SHA256 computed successfully.", "success")
                return render_template("verify.html", app_name=APP_NAME, upload_result=upload_result, run_result=run_result, runs=runs)

            if mode == "run":
                run_id = int(request.form.get("run_id") or "0")
                which = (request.form.get("which") or "input").strip().lower()

                run = RunHistory.query.filter_by(id=run_id, user_id=current_user.id).first()
                if not run:
                    flash("Run not found.", "danger")
                    return redirect(url_for("verify"))

                try:
                    findings = json.loads(run.findings_json or "{}")
                except Exception:
                    findings = {}

                filename = run.input_filename if which == "input" else (run.output_filename or "")
                if not filename:
                    flash("Output missing for this run.", "warning")
                    return redirect(url_for("verify"))

                full = safe_join(uploads, filename)
                if not full.exists():
                    flash("File missing from uploads folder.", "danger")
                    return redirect(url_for("verify"))

                sha_current = sha256_file(str(full))
                sha_recorded = None

                if which == "input":
                    sha_recorded = findings.get("checksums", {}).get("input_sha256")
                else:
                    sha_recorded = findings.get("checksums", {}).get("output_sha256")

                sha_recorded = (sha_recorded or "").strip().lower() or None
                run_result = {
                    "run_id": run.id,
                    "which": which,
                    "filename": filename,
                    "sha256_current": sha_current,
                    "sha256_recorded": sha_recorded,
                    "match": (sha_current == sha_recorded) if sha_recorded else None
                }
                flash("Run verification computed.", "success")
                return render_template("verify.html", app_name=APP_NAME, upload_result=upload_result, run_result=run_result, runs=runs)

            flash("Unknown verify mode.", "danger")
            return redirect(url_for("verify"))

        except Exception:
            traceback.print_exc()
            flash("Verification failed safely.", "danger")
            return redirect(url_for("verify"))

    # ---- Admin ----
    @app.route("/admin/users")
    @login_required
    @role_required("Admin")
    def admin_users():
        try:
            users = User.query.order_by(User.created_at.asc()).all()
            return render_template("admin_users.html", app_name=APP_NAME, users=users)
        except Exception:
            traceback.print_exc()
            flash("Admin users page failed safely.", "danger")
            return redirect(url_for("dashboard"))

    @app.route("/admin/users/<int:user_id>/role", methods=["POST"])
    @login_required
    @role_required("Admin")
    def admin_set_role(user_id: int):
        try:
            role = (request.form.get("role") or "Analyst").strip()
            if role not in ("Admin", "Analyst"):
                flash("Invalid role.", "danger")
                return redirect(url_for("admin_users"))

            user = db.session.get(User, user_id)
            if not user:
                flash("User not found.", "danger")
                return redirect(url_for("admin_users"))

            if user.role == "Admin" and role == "Analyst":
                admin_count = User.query.filter_by(role="Admin").count()
                if admin_count <= 1:
                    flash("Cannot demote the last Admin.", "warning")
                    return redirect(url_for("admin_users"))

            user.role = role
            db.session.commit()
            flash(f"Updated role for {user.username} → {role}", "success")
            return redirect(url_for("admin_users"))

        except Exception:
            traceback.print_exc()
            flash("Role update failed safely.", "danger")
            return redirect(url_for("admin_users"))

    @app.route("/admin/runs")
    @login_required
    @role_required("Admin")
    def admin_runs():
        try:
            runs = RunHistory.query.order_by(RunHistory.created_at.desc()).limit(500).all()
            return render_template("admin_runs.html", app_name=APP_NAME, runs=runs)
        except Exception:
            traceback.print_exc()
            flash("Admin runs page failed safely.", "danger")
            return redirect(url_for("dashboard"))

    @app.route("/admin/run/<int:run_id>")
    @login_required
    @role_required("Admin")
    def admin_run_detail(run_id: int):
        try:
            run = RunHistory.query.filter_by(id=run_id).first()
            if not run:
                abort(404)
            try:
                findings = json.loads(run.findings_json or "{}")
            except Exception:
                findings = {"note": "Could not parse findings JSON safely."}
            ledger_tail = read_ledger_tail(get_base_dir(), n=6)
            return render_template("run_detail.html", app_name=APP_NAME, run=run, findings=findings, ledger_tail=ledger_tail)
        except Exception:
            traceback.print_exc()
            flash("Admin run detail failed safely.", "danger")
            return redirect(url_for("admin_runs"))

    return app

if __name__ == "__main__":
    app = create_app()

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    # Open browser after small delay
    threading.Timer(1.5, open_browser).start()

    app.run(host="127.0.0.1", port=5000, debug=False)