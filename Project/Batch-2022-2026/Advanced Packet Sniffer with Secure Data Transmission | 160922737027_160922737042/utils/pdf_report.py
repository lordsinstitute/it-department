import io
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def build_pdf_report(run, summary: dict, findings: list, indicators: dict) -> io.BytesIO:
    """
    Generates a minimal but professional PDF report using ReportLab.
    Returns BytesIO for Flask send_file().
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    def header(title: str):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1 * inch, height - 1 * inch, title)
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 1.25 * inch, f"Generated: {datetime.utcnow().isoformat()}Z")

    def line(y, text, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(1 * inch, y, text)

    header("Packet Sniffer - Analysis Report")

    y = height - 1.7 * inch
    line(y, f"Run ID: {run.id}", True); y -= 14
    line(y, f"User: {run.user.username}"); y -= 14
    line(y, f"Created (UTC): {run.created_at.isoformat()}"); y -= 14
    line(y, f"Input Mode: {run.input_mode}"); y -= 14
    line(y, f"Source File: {run.source_file or 'N/A'}"); y -= 14
    line(y, f"Risk: {run.risk} | Score: {run.score}", True); y -= 14
    line(y, f"Evidence Hash (SHA256 Chain): {run.evidence_hash}", True); y -= 18
    line(y, f"Previous Hash: {run.prev_hash or 'GENESIS'}"); y -= 22

    # Summary block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Summary"); y -= 16
    c.setFont("Helvetica", 10)
    for k, v in (summary or {}).items():
        c.drawString(1 * inch, y, f"- {k}: {v}")
        y -= 14
        if y < 1.2 * inch:
            c.showPage()
            y = height - 1.2 * inch

    # Findings
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Findings"); y -= 16
    c.setFont("Helvetica", 10)

    if not findings:
        c.drawString(1 * inch, y, "No findings detected.")
        y -= 14
    else:
        for idx, f in enumerate(findings, start=1):
            title = f.get("title", "Finding")
            severity = f.get("severity", "Info")
            c.setFont("Helvetica-Bold", 10)
            c.drawString(1 * inch, y, f"{idx}. [{severity}] {title}")
            y -= 14
            c.setFont("Helvetica", 10)
            rec = f.get("recommendation", "")
            if rec:
                c.drawString(1 * inch, y, f"   Recommendation: {rec[:110]}")
                y -= 14

            details = f.get("details")
            if details:
                # print a compact representation
                text = str(details)
                text = text.replace("\n", " ")
                c.drawString(1 * inch, y, f"   Details: {text[:120]}")
                y -= 14

            y -= 6
            if y < 1.2 * inch:
                c.showPage()
                y = height - 1.2 * inch

    # Indicators
    y -= 4
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Top Indicators"); y -= 16
    c.setFont("Helvetica", 10)

    top_ips = indicators.get("top_ips", []) if indicators else []
    top_domains = indicators.get("top_domains", []) if indicators else []
    top_ports = indicators.get("top_ports", []) if indicators else []

    c.drawString(1 * inch, y, f"Top IPs: {top_ips}"); y -= 14
    c.drawString(1 * inch, y, f"Top Domains: {top_domains}"); y -= 14
    c.drawString(1 * inch, y, f"Top Ports: {top_ports}"); y -= 14

    c.showPage()
    c.save()

    buf.seek(0)
    return buf