import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def build_pdf_report(app_root: str, run, payload: dict) -> str:
    reports_dir = os.path.join(app_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    path = os.path.join(reports_dir, f"scan_report_{run.id}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    y = height - 2 * cm

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "REAL TIME PORT SCANNER AND DETECTOR - REPORT")
    y -= 1 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    y -= 0.7 * cm

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Summary")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    lines = [
        f"Run ID: {run.id}",
        f"Target(s): {run.target}",
        f"Scan Type: {run.scan_type}",
        f"Created By: {run.created_by}",
        f"Created At: {run.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Risk Level: {run.risk_level} (Score: {run.risk_score})",
        f"Evidence Chain Hash: {run.ledger_hash or 'N/A'}",
    ]
    for line in lines:
        c.drawString(2 * cm, y, line)
        y -= 0.5 * cm

    y -= 0.4 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Findings (Open Ports)")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)

    results = payload.get("results") or []
    for t in results:
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 10)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, y, f"Target: {t.get('target')}  |  Resolved IP: {t.get('resolved_ip')}")
        y -= 0.5 * cm

        c.setFont("Helvetica", 10)
        open_ports = t.get("open_ports") or []
        if not open_ports:
            c.drawString(2 * cm, y, "No open ports detected.")
            y -= 0.5 * cm
        else:
            for op in open_ports[:200]:
                if y < 2.5 * cm:
                    c.showPage()
                    y = height - 2 * cm
                    c.setFont("Helvetica", 10)
                banner = (op.get("banner") or "")
                banner = banner[:80] + ("..." if len(banner) > 80 else "")
                c.drawString(2 * cm, y, f"- Port {op.get('port')}  Banner: {banner}")
                y -= 0.45 * cm

        y -= 0.3 * cm

    # Alerts section
    alerts = payload.get("alerts") or []
    if alerts:
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Alert Simulation")
        y -= 0.6 * cm

        c.setFont("Helvetica", 10)
        for a in alerts[:100]:
            if y < 2.5 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 10)
            c.drawString(2 * cm, y, f"- {a.get('target')}:{a.get('port')} -> {a.get('message')}")
            y -= 0.45 * cm

    c.showPage()
    c.save()
    return path