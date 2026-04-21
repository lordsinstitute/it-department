import json
import time
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app
from flask_login import login_required, current_user
from detector.extensions import db
from models.scan import ScanRun

from utils.scanner import start_scan_job, get_job_status
from utils.risk import compute_risk
from utils.ledger import append_ledger_entry, ensure_ledger_exists
from utils.pdf_report import build_pdf_report

main_bp = Blueprint("main", __name__)

@main_bp.get("/")
@login_required
def dashboard():
    try:
        total = ScanRun.query.count()
        high_critical = ScanRun.query.filter(ScanRun.risk_level.in_(["High", "Critical"])).count()
        unique_targets = db.session.query(ScanRun.target).distinct().count()
        last = ScanRun.query.order_by(ScanRun.created_at.desc()).first()

        # Risk distribution
        risk_counts = {k: 0 for k in ["Low", "Medium", "High", "Critical"]}
        for k, v in db.session.query(ScanRun.risk_level, db.func.count(ScanRun.id)).group_by(ScanRun.risk_level).all():
            if k in risk_counts:
                risk_counts[k] = v

        # Scans per day (last 7 days)
        today = datetime.utcnow().date()
        labels = []
        values = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            labels.append(d.strftime("%b %d"))
            start = datetime(d.year, d.month, d.day)
            end = start + timedelta(days=1)
            cnt = ScanRun.query.filter(ScanRun.created_at >= start, ScanRun.created_at < end).count()
            values.append(cnt)

        return render_template(
            "dashboard.html",
            total=total,
            high_critical=high_critical,
            unique_targets=unique_targets,
            last_scan=last,
            risk_counts=risk_counts,
            scans_labels=json.dumps(labels),
            scans_values=json.dumps(values),
        )
    except Exception:
        flash("Dashboard could not load completely. Showing partial data.", "warning")
        return render_template(
            "dashboard.html",
            total=0, high_critical=0, unique_targets=0, last_scan=None,
            risk_counts={"Low":0,"Medium":0,"High":0,"Critical":0},
            scans_labels=json.dumps([]),
            scans_values=json.dumps([]),
        )

@main_bp.get("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")

@main_bp.post("/analyze")
@login_required
def analyze_post():
    """
    Analyze supports:
    - Text input: target + port mode
    - File upload: each line is a target (ip/hostname)
    """
    try:
        target_text = (request.form.get("target") or "").strip()
        mode = request.form.get("port_mode") or "common"
        ports_text = (request.form.get("ports") or "").strip()
        timeout_ms = int(request.form.get("timeout_ms") or "600")
        max_threads = int(request.form.get("max_threads") or "200")

        uploaded = request.files.get("target_file")
        targets = []

        if uploaded and uploaded.filename:
            # Parse file lines
            content = uploaded.read().decode(errors="ignore").splitlines()
            targets = [line.strip() for line in content if line.strip() and not line.strip().startswith("#")]
            if not targets:
                flash("Uploaded file has no valid targets.", "warning")
                return redirect(url_for("main.analyze"))
        else:
            if not target_text:
                flash("Provide a target (IP/hostname) or upload a target file.", "warning")
                return redirect(url_for("main.analyze"))
            targets = [target_text]

        job_id = start_scan_job(
            targets=targets,
            port_mode=mode,
            ports_text=ports_text,
            timeout_ms=timeout_ms,
            max_threads=max_threads,
            requested_by=current_user.username,
        )

        flash("Scan started. Progress updates are live on this page.", "info")
        return redirect(url_for("main.results_live", job_id=job_id))
    except Exception:
        flash("Could not start scan due to an unexpected error.", "danger")
        return redirect(url_for("main.analyze"))

@main_bp.get("/results/live/<job_id>")
@login_required
def results_live(job_id):
    # Page uses JS polling to show progress then redirects to saved result
    return render_template("results.html", job_id=job_id, run=None, live=True)

@main_bp.get("/api/job/<job_id>")
@login_required
def api_job(job_id):
    try:
        status = get_job_status(job_id)
        return jsonify(status)
    except Exception:
        return jsonify({"state": "error", "message": "Failed to fetch job status."}), 200

@main_bp.post("/api/job/<job_id>/finalize")
@login_required
def api_finalize(job_id):
    """
    When job finishes, persist it to DB + ledger.
    Frontend calls this once when state == done.
    """
    try:
        status = get_job_status(job_id)
        if status.get("state") != "done":
            return jsonify({"ok": False, "message": "Job not finished."}), 200

        payload = status.get("result") or {}
        # Compute risk
        risk_level, risk_score, risk_reasons = compute_risk(payload)

        run = ScanRun(
            target=payload.get("targets_display", "unknown"),
            scan_type=payload.get("scan_type", "TCP Connect"),
            findings_json=json.dumps(payload, ensure_ascii=False),
            risk_level=risk_level,
            risk_score=risk_score,
            created_by=current_user.username,
        )
        db.session.add(run)
        db.session.commit()

        # Evidence integrity ledger
        ensure_ledger_exists(current_app.config["LEDGER_PATH"])
        chain_hash = append_ledger_entry(
            ledger_path=current_app.config["LEDGER_PATH"],
            run_id=run.id,
            created_at=run.created_at.isoformat(),
            target=run.target,
            findings_json=run.findings_json,
        )
        run.ledger_hash = chain_hash
        db.session.commit()

        return jsonify({"ok": True, "run_id": run.id}), 200
    except Exception:
        return jsonify({"ok": False, "message": "Finalize failed."}), 200

@main_bp.get("/results/<int:run_id>")
@login_required
def results_saved(run_id):
    try:
        run = ScanRun.query.get_or_404(run_id)
        payload = json.loads(run.findings_json or "{}")
        return render_template("results.html", job_id=None, run=run, payload=payload, live=False)
    except Exception:
        flash("Could not load results.", "danger")
        return redirect(url_for("main.history"))

@main_bp.get("/history")
@login_required
def history():
    try:
        runs = ScanRun.query.order_by(ScanRun.created_at.desc()).limit(200).all()
        return render_template("history.html", runs=runs)
    except Exception:
        flash("History could not load completely.", "warning")
        return render_template("history.html", runs=[])

@main_bp.get("/report/<int:run_id>")
@login_required
def report(run_id):
    """
    PDF report export (ReportLab).
    """
    try:
        run = ScanRun.query.get_or_404(run_id)
        payload = json.loads(run.findings_json or "{}")

        pdf_path = build_pdf_report(
            app_root=current_app.config["APP_ROOT"],
            run=run,
            payload=payload
        )
        return send_file(pdf_path, as_attachment=True, download_name=f"scan_report_{run_id}.pdf")
    except Exception:
        flash("Could not generate PDF report.", "danger")
        return redirect(url_for("main.results_saved", run_id=run_id))