import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Lightbulb, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { askAI } from "@/lib/api";

const SUSPICIOUS_KEYWORDS = [
  "union", "select", "drop", "delete", "exec", "shell", "wget", "curl",
  "script", "alert", "onerror", "eval", "nmap", "brute", "inject",
  "malicious", "attack", "exploit", "backdoor", "payload", "overflow",
  "unauthorized", "forbidden", "denied", "failed", "critical", "warning",
  "suspicious", "phishing", "ransomware", "trojan", "keylogger", "rootkit",
  "rm -rf", "chmod", "passwd", "sudo", "ssh", "telnet",
];

const IP_REGEX = /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g;

const detectSeverity = (line: string): "Critical" | "High" | "Medium" | "Low" => {
  const l = line.toLowerCase();
  if (["critical", "exec", "shell", "drop", "rm -rf", "backdoor", "exploit"].some((k) => l.includes(k))) return "Critical";
  if (["warning", "alert", "union", "select", "script", "nmap", "brute"].some((k) => l.includes(k))) return "High";
  if (["suspicious", "failed", "denied", "forbidden", "unauthorized"].some((k) => l.includes(k))) return "Medium";
  return "Low";
};

const detectAttackType = (line: string): string => {
  const l = line.toLowerCase();
  if (["union", "select", "or '1'='1", "drop table"].some((k) => l.includes(k))) return "SQL Injection";
  if (["<script>", "onerror", "eval(", "alert("].some((k) => l.includes(k))) return "XSS";
  if (["exec", "shell", "wget", "curl", "bash", "rm -rf"].some((k) => l.includes(k))) return "Command Injection";
  if (["nmap", "scan", "port"].some((k) => l.includes(k))) return "Port Scan";
  if (["brute", "failed ssh", "failed login"].some((k) => l.includes(k))) return "Brute Force";
  if (["phishing", "spoof"].some((k) => l.includes(k))) return "Phishing";
  if (l.includes("info") && !["warning", "alert", "critical", "error"].some((k) => l.includes(k))) return "Normal";
  return "Suspicious";
};

const highlightLine = (line: string) => {
  const parts: { text: string; suspicious: boolean }[] = [];
  let remaining = line;

  while (remaining.length > 0) {
    let earliest = -1;
    let matchWord = "";
    for (const kw of SUSPICIOUS_KEYWORDS) {
      const idx = remaining.toLowerCase().indexOf(kw);
      if (idx !== -1 && (earliest === -1 || idx < earliest)) {
        earliest = idx;
        matchWord = remaining.slice(idx, idx + kw.length);
      }
    }
    if (earliest === -1) {
      parts.push({ text: remaining, suspicious: false });
      break;
    }
    if (earliest > 0) parts.push({ text: remaining.slice(0, earliest), suspicious: false });
    parts.push({ text: matchWord, suspicious: true });
    remaining = remaining.slice(earliest + matchWord.length);
  }

  return parts.map((p, i) =>
    p.suspicious ? (
      <span key={i} className="text-critical font-bold bg-critical/10 px-0.5 rounded">{p.text}</span>
    ) : (
      <span key={i}>{p.text}</span>
    )
  );
};

const sevBadge: Record<string, string> = {
  Critical: "bg-critical/15 text-critical",
  High: "bg-high/15 text-high",
  Medium: "bg-warning/15 text-warning",
  Low: "bg-success/15 text-success",
};

interface LogLineViewerProps {
  rawInput: string;
}

const LogLineViewer = ({ rawInput }: LogLineViewerProps) => {
  const [expandedLine, setExpandedLine] = useState<number | null>(null);
  const [explanations, setExplanations] = useState<Record<number, string>>({});
  const [loadingLine, setLoadingLine] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);

  const lines = rawInput
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) return null;

  const parsedLines = lines.map((line) => {
    const ips = line.match(IP_REGEX) || [];
    const severity = detectSeverity(line);
    const attackType = detectAttackType(line);
    return { line, ips, severity, attackType };
  });

  const suspiciousCount = parsedLines.filter((p) => p.severity !== "Low").length;
  const displayed = showAll ? parsedLines : parsedLines.slice(0, 8);

  const explainLine = async (idx: number, line: string) => {
    if (explanations[idx]) {
      setExpandedLine(expandedLine === idx ? null : idx);
      return;
    }
    setLoadingLine(idx);
    setExpandedLine(idx);
    try {
      const answer = await askAI(
        `Explain this single log line from a security perspective. What does it mean, is it dangerous, and what should an admin do?\n\nLog line: "${line}"\n\nBe concise (3-4 sentences).`
      );
      setExplanations((prev) => ({ ...prev, [idx]: answer }));
    } catch (err: any) {
      setExplanations((prev) => ({ ...prev, [idx]: `⚠️ ${err.message}` }));
    } finally {
      setLoadingLine(null);
    }
  };

  return (
    <motion.div
      className="glass-card rounded-xl p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">Log Line Analysis</h3>
        <div className="flex items-center gap-2">
          {suspiciousCount > 0 && (
            <span className="rounded-full bg-critical/15 px-2.5 py-0.5 text-[10px] font-bold text-critical">
              {suspiciousCount} suspicious {suspiciousCount === 1 ? "line" : "lines"}
            </span>
          )}
          <span className="text-[10px] text-muted-foreground">{lines.length} lines total</span>
        </div>
      </div>

      <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
        {displayed.map((p, idx) => (
          <div key={idx}>
            <div
              className={`flex items-start gap-2 rounded-lg border px-3 py-2 transition-colors ${
                p.severity === "Critical"
                  ? "border-critical/30 bg-critical/5"
                  : p.severity === "High"
                  ? "border-high/20 bg-high/5"
                  : "border-border bg-secondary/20"
              }`}
            >
              <span className="shrink-0 w-5 text-right font-mono text-[10px] text-muted-foreground/50 mt-0.5">
                {idx + 1}
              </span>
              <div className="flex-1 min-w-0">
                <p className="font-mono text-[11px] text-foreground/90 break-all leading-relaxed">
                  {highlightLine(p.line)}
                </p>
                <div className="flex items-center gap-2 mt-1.5">
                  {p.ips.map((ip, i) => (
                    <span key={i} className="font-mono text-[9px] text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                      {ip}
                    </span>
                  ))}
                  <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${sevBadge[p.severity]}`}>
                    {p.severity}
                  </span>
                  {p.attackType !== "Normal" && (
                    <span className="text-[9px] text-muted-foreground bg-secondary/50 px-1.5 py-0.5 rounded">
                      {p.attackType}
                    </span>
                  )}
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="shrink-0 h-6 px-2 text-[9px] text-primary hover:bg-primary/10 gap-1"
                onClick={() => explainLine(idx, p.line)}
                disabled={loadingLine === idx}
              >
                {loadingLine === idx ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Lightbulb className="h-3 w-3" />
                )}
                Explain
              </Button>
            </div>

            <AnimatePresence>
              {expandedLine === idx && explanations[idx] && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="ml-7 mr-2 mb-1"
                >
                  <div className="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 mt-1">
                    <p className="text-[11px] text-foreground/80 leading-relaxed whitespace-pre-wrap">
                      {explanations[idx]}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>

      {parsedLines.length > 8 && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="flex items-center gap-1 mt-3 text-[10px] text-primary hover:underline mx-auto"
        >
          {showAll ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {showAll ? "Show less" : `Show all ${parsedLines.length} lines`}
        </button>
      )}
    </motion.div>
  );
};

export default LogLineViewer;
