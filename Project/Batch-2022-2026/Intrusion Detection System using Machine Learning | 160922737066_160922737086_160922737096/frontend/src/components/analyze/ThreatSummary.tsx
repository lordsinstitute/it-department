import { motion } from "framer-motion";
import { ShieldAlert, Download, FileJson, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { generatePDFReport } from "@/lib/pdf-export";
import type { StoredAlert } from "@/context/AlertContext";

interface ThreatSummaryProps {
  result: { prediction: string; risk_score: number; explanation: string; best_model?: string; models?: Record<string, { accuracy: number; risk_score: number }> };
  userInput: string;
  alerts: StoredAlert[];
}

const ThreatSummary = ({ result, userInput, alerts }: ThreatSummaryProps) => {
  const severity = result.risk_score >= 80 ? "Critical" : result.risk_score >= 60 ? "High" : result.risk_score >= 30 ? "Medium" : "Low";

  const lines = userInput.split("\n").filter((l) => l.trim());
  const suspiciousKeywords = ["union", "select", "drop", "exec", "script", "nmap", "brute", "shell", "wget", "backdoor", "exploit"];
  const suspiciousLines = lines.filter((l) => suspiciousKeywords.some((k) => l.toLowerCase().includes(k)));

  const exportJSON = () => {
    const data = {
      timestamp: new Date().toISOString(),
      prediction: result.prediction,
      risk_score: result.risk_score,
      severity,
      explanation: result.explanation,
      input_lines: lines.length,
      suspicious_lines: suspiciousLines.length,
      raw_input: userInput.slice(0, 2000),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `threat_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sevColor: Record<string, string> = {
    Critical: "text-critical",
    High: "text-high",
    Medium: "text-warning",
    Low: "text-success",
  };

  return (
    <motion.div
      className="glass-card rounded-xl p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Threat Summary</h3>
        </div>
        <div className="flex gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={exportJSON}
            className="h-7 text-[10px] gap-1.5 border-border"
          >
            <FileJson className="h-3 w-3" />
            JSON
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => generatePDFReport(alerts.slice(0, 20))}
            className="h-7 text-[10px] gap-1.5 border-border"
          >
            <FileText className="h-3 w-3" />
            PDF
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="rounded-lg border border-border bg-secondary/30 px-3 py-4 text-center min-w-0">
          <p className={`text-base sm:text-lg font-bold font-mono truncate ${sevColor[severity]}`}>{severity}</p>
          <p className="text-[9px] text-muted-foreground mt-1">Severity</p>
        </div>
        <div className="rounded-lg border border-border bg-secondary/30 px-3 py-4 text-center min-w-0">
          <p className="text-base sm:text-lg font-bold font-mono text-foreground">{result.risk_score}</p>
          <p className="text-[9px] text-muted-foreground mt-1">Risk Score</p>
        </div>
        <div className="rounded-lg border border-border bg-secondary/30 px-3 py-4 text-center min-w-0">
          <p className="text-base sm:text-lg font-bold font-mono text-foreground">{lines.length}</p>
          <p className="text-[9px] text-muted-foreground mt-1">Lines Analyzed</p>
        </div>
        <div className="rounded-lg border border-border bg-secondary/30 px-3 py-4 text-center min-w-0">
          <p className={`text-base sm:text-lg font-bold font-mono ${suspiciousLines.length > 0 ? "text-critical" : "text-success"}`}>
            {suspiciousLines.length}
          </p>
          <p className="text-[9px] text-muted-foreground mt-1">Suspicious</p>
        </div>
        {result.best_model && (
          <div className="rounded-lg border border-primary/40 bg-primary/5 px-3 py-4 text-center min-w-0">
            <p className="text-[10px] sm:text-xs font-bold font-mono text-primary leading-tight truncate">{result.best_model}</p>
            <p className="text-[9px] text-muted-foreground mt-1">Best Model</p>
          </div>
        )}
        {result.models && (
          <div className="rounded-lg border border-border bg-secondary/30 px-3 py-4 text-center min-w-0">
            <p className="text-base sm:text-lg font-bold font-mono text-foreground">{Object.keys(result.models).length}</p>
            <p className="text-[9px] text-muted-foreground mt-1">Models Compared</p>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ThreatSummary;
