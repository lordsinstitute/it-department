from io import BytesIO
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file
from sqlalchemy import func

from . import db
from .models import ScanResult
from .auth import login_required
from .analysis import analyze_email_content
from .ml_engine import load_metrics, model_exists
from utils.helpers import allowed_file, safe_read_uploaded_file
from utils.evidence import append_to_ledger
from utils.reporting import generate_pdf_report

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("auth.login"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    try:
        user_id = session["user_id"]

        total_scans = ScanResult.query.filter_by(user_id=user_id).count()
        threat_count = ScanResult.query.filter(
            ScanResult.user_id == user_id,
            ScanResult.prediction_label.in_(["Spam", "Scam"])
        ).count()
        critical_count = ScanResult.query.filter_by(user_id=user_id, risk_level="Critical").count()
        avg_risk = db.session.query(func.avg(ScanResult.risk_score)).filter_by(user_id=user_id).scalar() or 0

        recent = ScanResult.query.filter_by(user_id=user_id).order_by(ScanResult.created_at.desc()).limit(7).all()

        risk_labels = ["Low", "Medium", "High", "Critical"]
        risk_counts = [ScanResult.query.filter_by(user_id=user_id, risk_level=label).count() for label in risk_labels]

        label_counts = {
            "Safe": ScanResult.query.filter_by(user_id=user_id, prediction_label="Safe").count(),
            "Spam": ScanResult.query.filter_by(user_id=user_id, prediction_label="Spam").count(),
            "Scam": ScanResult.query.filter_by(user_id=user_id, prediction_label="Scam").count()
        }

        trend_labels = [row.created_at.strftime("%d-%b") for row in reversed(recent)]
        trend_scores = [row.risk_score for row in reversed(recent)]

        metrics = load_metrics()

        return render_template(
            "dashboard.html",
            total_scans=total_scans,
            threat_count=threat_count,
            critical_count=critical_count,
            avg_risk=round(avg_risk, 2),
            risk_labels=risk_labels,
            risk_counts=risk_counts,
            label_counts=label_counts,
            trend_labels=trend_labels,
            trend_scores=trend_scores,
            metrics=metrics,
            recent=recent,
            model_ready=model_exists()
        )
    except Exception:
        flash("Dashboard could not be fully loaded.", "warning")
        return render_template(
            "dashboard.html",
            total_scans=0,
            threat_count=0,
            critical_count=0,
            avg_risk=0,
            risk_labels=["Low", "Medium", "High", "Critical"],
            risk_counts=[0, 0, 0, 0],
            label_counts={"Safe": 0, "Spam": 0, "Scam": 0},
            trend_labels=[],
            trend_scores=[],
            metrics=load_metrics(),
            recent=[],
            model_ready=model_exists()
        )


@main_bp.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "GET":
        return render_template("analyze.html", model_ready=model_exists())

    try:
        if not model_exists():
            flash("Model not trained yet. Run: python train_model.py", "danger")
            return render_template("analyze.html", model_ready=False)

        input_mode = request.form.get("input_mode", "text")
        text_input = request.form.get("email_text", "").strip()
        uploaded = request.files.get("email_file")

        source_type = "text"
        original_filename = None
        final_text = ""

        if input_mode == "file":
            source_type = "file"
            if not uploaded or uploaded.filename == "":
                flash("Please choose a .txt or .eml file.", "warning")
                return render_template("analyze.html", model_ready=True)

            if not allowed_file(uploaded.filename):
                flash("Only .txt and .eml files are allowed.", "danger")
                return render_template("analyze.html", model_ready=True)

            original_filename = uploaded.filename
            final_text = safe_read_uploaded_file(uploaded)
        else:
            final_text = text_input

        if not final_text.strip():
            flash("Please provide email text or upload a valid file.", "warning")
            return render_template("analyze.html", model_ready=True)

        analysis = analyze_email_content(final_text)

        scan = ScanResult(
            source_type=source_type,
            original_filename=original_filename,
            input_text=final_text,
            cleaned_text=analysis["cleaned_text"],
            ml_label=analysis["ml_label"],
            prediction_label=analysis["prediction_label"],
            confidence_score=analysis["confidence_score"],
            risk_score=analysis["risk_score"],
            risk_level=analysis["risk_level"],
            suspicious_links=analysis["suspicious_links"],
            urgent_words=analysis["urgent_words"],
            financial_words=analysis["financial_words"],
            attachment_words=analysis["attachment_words"],
            impersonation_words=analysis["impersonation_words"],
            summary=analysis["summary"],
            finding_details=analysis["finding_details"],
            user_id=session["user_id"]
        )

        db.session.add(scan)
        db.session.commit()

        evidence_hash = append_to_ledger(scan)
        scan.evidence_hash = evidence_hash
        db.session.commit()

        if scan.risk_level in ("High", "Critical"):
            print(f"[ALERT SIMULATION] Threat detected -> Scan ID {scan.id} | Final Label {scan.prediction_label} | Level {scan.risk_level}")

        flash("Analysis completed successfully.", "success")
        return redirect(url_for("main.result", scan_id=scan.id))

    except FileNotFoundError:
        flash("Model files are missing. Run: python train_model.py", "danger")
        return render_template("analyze.html", model_ready=False)
    except Exception as exc:
        db.session.rollback()
        print("\n===== FULL ERROR TRACE =====")
        traceback.print_exc()
        print("===== END TRACE =====\n")

        flash(f"Error: {str(exc)}", "danger")
        return render_template("analyze.html", model_ready=model_exists())


@main_bp.route("/result/<int:scan_id>")
@login_required
def result(scan_id):
    try:
        scan = ScanResult.query.filter_by(id=scan_id, user_id=session["user_id"]).first_or_404()
        return render_template("result.html", scan=scan)
    except Exception:
        flash("Result page could not be loaded.", "danger")
        return redirect(url_for("main.history"))


@main_bp.route("/history")
@login_required
def history():
    try:
        scans = ScanResult.query.filter_by(user_id=session["user_id"]).order_by(ScanResult.created_at.desc()).all()
        return render_template("history.html", scans=scans)
    except Exception:
        flash("History could not be loaded.", "danger")
        return render_template("history.html", scans=[])


@main_bp.route("/download-report/<int:scan_id>")
@login_required
def download_report(scan_id):
    try:
        scan = ScanResult.query.filter_by(id=scan_id, user_id=session["user_id"]).first_or_404()
        pdf_data = generate_pdf_report(scan)
        filename = f"email_shield_report_{scan.id}.pdf"
        return send_file(
            BytesIO(pdf_data),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as exc:
        print(f"[PDF ERROR] {exc}")
        flash("Report generation failed.", "danger")
        return redirect(url_for("main.result", scan_id=scan_id))