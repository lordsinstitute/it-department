from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def _line(c: canvas.Canvas, x: float, y: float, text: str, size: int = 10):
    c.setFont("Helvetica", size)
    c.drawString(x, y, text)


def build_pdf_report(pdf_path: Path, run, analysis: dict) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4

    margin = 2.0 * cm
    y = h - margin

    c.setTitle(f"DDoS Packet Flow Report - Run {run.id}")

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "DDoS Detection Report (Packet Flow Analysis)")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.5 * cm
    c.drawString(margin, y, f"Run ID: {run.id} | User: {run.user.username} | Source: {run.source}")
    y -= 0.5 * cm
    c.drawString(margin, y, f"Risk Level: {run.risk_level} | Risk Score: {run.risk_score}")
    y -= 0.8 * cm

    # Evidence
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Evidence Integrity (Local Ledger)")
    y -= 0.6 * cm
    _line(c, margin, y, f"Chain Index: {run.chain_index if run.chain_index is not None else 'N/A'}")
    y -= 0.45 * cm
    _line(c, margin, y, f"Prev Hash : {run.prev_hash or 'N/A'}", size=9)
    y -= 0.45 * cm
    _line(c, margin, y, f"Hash      : {run.evidence_hash or 'N/A'}", size=9)
    y -= 0.8 * cm

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Traffic Summary")
    y -= 0.6 * cm

    victim = analysis.get("victim_candidate") or {}
    _line(c, margin, y, f"Total Lines: {analysis.get('total_lines', 0)}")
    y -= 0.45 * cm
    _line(c, margin, y, f"Total Packets Parsed: {analysis.get('total_packets', 0)}")
    y -= 0.45 * cm
    _line(c, margin, y, f"Estimated Time Window (s): {analysis.get('time_window_seconds_est')}")
    y -= 0.45 * cm
    _line(c, margin, y, f"Victim Candidate: {victim.get('dst_ip')} | Packets: {victim.get('packets')} | Unique Sources: {victim.get('unique_sources')}")
    y -= 0.8 * cm

    # Findings
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Key Findings")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    findings = analysis.get("findings") or []
    if not findings:
        findings = ["No findings available."]

    for f in findings[:18]:
        if y < margin + 2 * cm:
            c.showPage()
            y = h - margin
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Key Findings (continued)")
            y -= 0.6 * cm
            c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"- {f}")
        y -= 0.45 * cm

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, margin * 0.6, "This report is generated locally. No external services were used.")
    c.save()