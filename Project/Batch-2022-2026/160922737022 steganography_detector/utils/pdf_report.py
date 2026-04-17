from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

def generate_pdf(scan):
    path = f"report_{scan.id}.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    c.drawString(50, 800, "Steganography Detection Report")
    c.drawString(50, 770, f"File: {scan.filename}")
    c.drawString(50, 750, f"Risk: {scan.risk}")
    c.drawString(50, 730, f"Hash: {scan.hash_value}")
    c.drawString(50, 710, f"Result: {scan.result}")
    c.save()
    return path