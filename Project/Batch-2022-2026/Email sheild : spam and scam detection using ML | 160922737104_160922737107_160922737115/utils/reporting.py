from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def wrap_text(text: str, width: int):
    words = (text or "").split()
    lines = []
    current = []

    for word in words:
        attempt = " ".join(current + [word])
        if len(attempt) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines or [""]


def generate_pdf_report(scan) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 20 * mm

    def line(text, step=6):
        nonlocal y
        c.drawString(18 * mm, y, str(text)[:140])
        y -= step * mm
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
            c.setFont("Helvetica", 10)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(18 * mm, y, "Email Shield - Scan Report")
    y -= 12 * mm

    c.setFont("Helvetica", 10)
    line(f"Scan ID: {scan.id}")
    line(f"Date: {scan.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    line(f"Source Type: {scan.source_type}")
    line(f"Original Filename: {scan.original_filename or 'N/A'}")
    line(f"ML Label: {scan.ml_label}")
    line(f"Final Prediction: {scan.prediction_label}")
    line(f"Confidence Score: {scan.confidence_score}%")
    line(f"Risk Score: {scan.risk_score}/100")
    line(f"Risk Level: {scan.risk_level}")
    line(f"Suspicious Links: {scan.suspicious_links}")
    line(f"Urgency Indicators: {scan.urgent_words}")
    line(f"Financial Keywords: {scan.financial_words}")
    line(f"Attachment Keywords: {scan.attachment_words}")
    line(f"Impersonation Indicators: {scan.impersonation_words}")
    line(f"Evidence Hash: {scan.evidence_hash}")

    y -= 2 * mm
    c.setFont("Helvetica-Bold", 12)
    line("Summary", 7)
    c.setFont("Helvetica", 10)
    for row in wrap_text(scan.summary, 110):
        line(row, 5)

    y -= 2 * mm
    c.setFont("Helvetica-Bold", 12)
    line("Finding Details", 7)
    c.setFont("Helvetica", 10)
    for row in wrap_text(scan.finding_details, 110):
        line(row, 5)

    y -= 2 * mm
    c.setFont("Helvetica-Bold", 12)
    line("Input Preview", 7)
    c.setFont("Helvetica", 10)
    for row in wrap_text(scan.cleaned_text[:900], 110):
        line(row, 5)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes