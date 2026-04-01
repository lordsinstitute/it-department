import json
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename
from detector import db
from detector.models import AnalysisRun
from detector.analyzer import analyze_input, extract_text_from_upload
from utils.ledger import append_to_ledger
from utils.report_generator import generate_pdf_report
from config import Config
from utils.helpers import safe_json_dumps

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def home():
    return redirect(url_for("auth.login"))

@main_bp.route("/dashboard")
@login_required
def dashboard():
    try:
        runs = AnalysisRun.query.order_by(AnalysisRun.created_at.desc()).limit(200).all()

        total = len(runs)
        by_level = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        for r in runs:
            by_level[r.risk_level] = by_level.get(r.risk_level, 0) + 1

        avg_score = int(sum([r.risk_score for r in runs]) / total) if total else 0

        last10 = list(reversed(runs[:10]))
        line_labels = [r.created_at.strftime("%H:%M") for r in last10]
        line_scores = [r.risk_score for r in last10]

        pie_labels = list(by_level.keys())
        pie_values = [by_level[k] for k in pie_labels]

        summary_cards = [
            {"title": "Total Analyses", "value": total},
            {"title": "Average Risk Score", "value": avg_score},
            {"title": "High + Critical", "value": by_level.get("High", 0) + by_level.get("Critical", 0)},
        ]

        return render_template(
            "dashboard.html",
            summary_cards=summary_cards,
            by_level=by_level,
            line_labels=json.dumps(line_labels),
            line_scores=json.dumps(line_scores),
            pie_labels=json.dumps(pie_labels),
            pie_values=json.dumps(pie_values),
        )
    except Exception:
        flash("Dashboard loaded with limited data due to a minor issue.", "warning")
        return render_template(
            "dashboard.html",
            summary_cards=[{"title": "Total Analyses", "value": 0},
                           {"title": "Average Risk Score", "value": 0},
                           {"title": "High + Critical", "value": 0}],
            by_level={"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            line_labels="[]",
            line_scores="[]",
            pie_labels='["Low","Medium","High","Critical"]',
            pie_values="[0,0,0,0]",
        )

@main_bp.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    # POST
    try:
        input_text = (request.form.get("input_text") or "").strip()
        upload = request.files.get("upload_file")

        used_type = None
        filename = None
        full_text = ""

        if upload and upload.filename:
            used_type = "file"
            filename = secure_filename(upload.filename)
            save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            upload.save(save_path)
            full_text = extract_text_from_upload(save_path)
        else:
            used_type = "text"
            full_text = input_text

        if not full_text.strip():
            flash("Provide text OR upload a file to analyze.", "warning")
            return redirect(url_for("main.analyze"))

        result = analyze_input(full_text)

        # Ledger append (blockchain-inspired)
        ledger_entry = {
            "score": result["score"],
            "level": result["level"],
            "summary": result["summary"],
            "input_type": used_type,
            "filename": filename,
            "snippet": (full_text[:180] + "...") if len(full_text) > 180 else full_text,
            "findings": result.get("findings", []),
            "intel_hits": result.get("intel_hits", []),
            "timestamp": result["timestamp"],
        }
        ledger_hash, prev_hash = append_to_ledger(ledger_entry)

        run = AnalysisRun(
            input_type=used_type,
            filename=filename,
            snippet=(full_text[:220] + "...") if len(full_text) > 220 else full_text,
            findings_json=safe_json_dumps(result),
            risk_score=int(result["score"]),
            risk_level=result["level"],
            ledger_hash=ledger_hash,
            ledger_prev_hash=prev_hash,
        )
        db.session.add(run)
        db.session.commit()

        # Alert Simulation
        if result["level"] in ("High", "Critical"):
            print(f"[ALERT SIMULATION] {result['level']} risk detected | Score={result['score']} | RunID={run.id}")
            flash(f"ALERT SIMULATION: {result['level']} risk detected (Score {result['score']}).", "danger")
        else:
            flash(f"Analysis complete. Risk: {result['level']} (Score {result['score']}).", "success")

        return redirect(url_for("main.run_detail", run_id=run.id))

    except Exception:
        flash("Analysis failed safely. Please try again.", "danger")
        return redirect(url_for("main.analyze"))

@main_bp.route("/history")
@login_required
def history():
    try:
        runs = AnalysisRun.query.order_by(AnalysisRun.created_at.desc()).limit(500).all()
        return render_template("history.html", runs=runs)
    except Exception:
        flash("History unavailable due to a minor issue.", "warning")
        return render_template("history.html", runs=[])

@main_bp.route("/run/<int:run_id>")
@login_required
def run_detail(run_id: int):
    try:
        run = AnalysisRun.query.get(run_id)
        if not run:
            flash("Run not found.", "warning")
            return redirect(url_for("main.history"))

        details = json.loads(run.findings_json) if run.findings_json else {}
        return render_template("run_detail.html", run=run, details=details)
    except Exception:
        flash("Unable to open run details (safe mode).", "warning")
        return redirect(url_for("main.history"))

@main_bp.route("/run/<int:run_id>/report")
@login_required
def download_report(run_id: int):
    try:
        run = AnalysisRun.query.get(run_id)
        if not run:
            flash("Run not found.", "warning")
            return redirect(url_for("main.history"))

        details = json.loads(run.findings_json) if run.findings_json else {}
        pdf_path = generate_pdf_report(run, details)

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"threat_report_run_{run_id}.pdf",
            mimetype="application/pdf",
        )
    except Exception:
        flash("Report generation failed safely. Try again.", "danger")
        return redirect(url_for("main.run_detail", run_id=run_id))