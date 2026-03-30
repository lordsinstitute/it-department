import jsPDF from "jspdf";
import type { StoredAlert } from "@/context/AlertContext";

const getSeverityLabel = (score: number) =>
  score >= 80 ? "Critical" : score >= 60 ? "High" : score >= 30 ? "Medium" : "Low";

export const generatePDFReport = (alerts: StoredAlert[]) => {
  const doc = new jsPDF();
  const now = new Date().toLocaleString();
  const pageWidth = doc.internal.pageSize.getWidth();

  // ── Header ──
  doc.setFillColor(6, 13, 31);
  doc.rect(0, 0, pageWidth, 40, "F");

  doc.setFontSize(22);
  doc.setTextColor(0, 245, 212);
  doc.text("IDS Shield", 14, 18);

  doc.setFontSize(10);
  doc.setTextColor(200, 200, 200);
  doc.text("Threat Analysis Report", 14, 26);

  doc.setFontSize(8);
  doc.setTextColor(150, 150, 150);
  doc.text(`Generated: ${now}`, 14, 34);
  doc.text(`Total Alerts: ${alerts.length}`, pageWidth - 14, 34, { align: "right" });

  // ── Summary stats ──
  const critical = alerts.filter((a) => a.severity === "Critical").length;
  const high = alerts.filter((a) => a.severity === "High").length;
  const medium = alerts.filter((a) => a.severity === "Medium").length;
  const low = alerts.filter((a) => a.severity === "Low").length;

  let y = 50;
  doc.setFontSize(11);
  doc.setTextColor(40, 40, 40);
  doc.text("Summary", 14, y);
  y += 8;

  const stats = [
    { label: "Critical", count: critical, r: 255, g: 69, b: 96 },
    { label: "High", count: high, r: 255, g: 140, b: 0 },
    { label: "Medium", count: medium, r: 255, g: 193, b: 7 },
    { label: "Low", count: low, r: 34, g: 197, b: 94 },
  ];

  stats.forEach((s, i) => {
    const x = 14 + i * 46;
    doc.setFillColor(s.r, s.g, s.b);
    doc.roundedRect(x, y, 40, 18, 2, 2, "F");
    doc.setFontSize(14);
    doc.setTextColor(255, 255, 255);
    doc.text(String(s.count), x + 20, y + 8, { align: "center" });
    doc.setFontSize(7);
    doc.text(s.label, x + 20, y + 14, { align: "center" });
  });

  y += 28;

  // ── Detailed alerts ──
  for (const a of alerts) {
    // Check if we need a new page (each alert needs ~55px)
    if (y > 230) {
      doc.addPage();
      y = 20;
    }

    // Alert card background
    doc.setFillColor(245, 245, 250);
    doc.roundedRect(14, y, pageWidth - 28, 50, 2, 2, "F");

    // Severity color bar
    const sevColors: Record<string, [number, number, number]> = {
      Critical: [255, 69, 96],
      High: [255, 140, 0],
      Medium: [255, 193, 7],
      Low: [34, 197, 94],
    };
    const [sr, sg, sb] = sevColors[a.severity] || [100, 100, 100];
    doc.setFillColor(sr, sg, sb);
    doc.roundedRect(14, y, 4, 50, 2, 2, "F");

    // Alert ID + timestamp
    doc.setFontSize(9);
    doc.setTextColor(0, 245, 212);
    doc.text(a.id, 22, y + 8);
    doc.setTextColor(130, 130, 130);
    doc.text(a.timestamp, pageWidth - 18, y + 8, { align: "right" });

    // Prediction + severity + score
    doc.setFontSize(11);
    doc.setTextColor(30, 30, 30);
    doc.text(`Prediction: ${a.prediction}`, 22, y + 18);

    doc.setFontSize(9);
    doc.setTextColor(sr, sg, sb);
    doc.text(`Severity: ${a.severity}`, 22, y + 26);

    doc.setTextColor(60, 60, 60);
    doc.text(`Risk Score: ${a.risk_score}/100`, 80, y + 26);
    doc.text(`Source IP: ${a.sourceIP}`, 130, y + 26);

    // Explanation (AI Reasoning)
    doc.setFontSize(8);
    doc.setTextColor(80, 80, 80);
    const explanationLines = doc.splitTextToSize(`AI Reasoning: ${a.explanation}`, pageWidth - 42);
    doc.text(explanationLines.slice(0, 2), 22, y + 35);

    y += 56;
  }

  // ── Footer on last page ──
  doc.setFontSize(7);
  doc.setTextColor(150, 150, 150);
  doc.text("IDS Shield — Adaptive Intrusion Detection System | Powered by ML Backend", 14, 288);
  doc.text(`Report generated: ${now}`, pageWidth - 14, 288, { align: "right" });

  doc.save(`IDS_Report_${Date.now()}.pdf`);
};
