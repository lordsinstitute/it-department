import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def build_pdf_report(output_path: str, app_name: str, run, findings: dict) -> None:
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 2.0 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2.0 * cm, y, f"{app_name} - Encoding Report")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2.0 * cm, y, f"Generated (UTC): {datetime.utcnow().isoformat(timespec='seconds')}Z")
    y -= 0.8 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.0 * cm, y, "Run Summary")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    lines = [
        f"Run ID: {run.id}",
        f"User ID: {run.user_id}",
        f"Type: {run.run_type}",
        f"Media: {run.media_type}",
        f"Input File: {run.input_filename}",
        f"Output File: {run.output_filename or '-'}",
        f"Risk: {run.risk_level} (score {run.risk_score})",
        f"Created (UTC): {run.created_at.isoformat(timespec='seconds')}Z",
        f"Ledger Hash: {run.ledger_hash}",
        f"Prev Hash: {run.ledger_prev_hash or '-'}",
    ]
    for line in lines:
        c.drawString(2.0 * cm, y, line)
        y -= 0.45 * cm

    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.0 * cm, y, "Encoding Findings (JSON)")
    y -= 0.6 * cm

    c.setFont("Helvetica", 9)
    findings_text = json.dumps(findings, indent=2, ensure_ascii=False)
    for raw_line in findings_text.splitlines():
        if y < 2.0 * cm:
            c.showPage()
            y = height - 2.0 * cm
            c.setFont("Helvetica", 9)
        c.drawString(2.0 * cm, y, raw_line[:150])
        y -= 0.38 * cm

    c.showPage()
    c.save()