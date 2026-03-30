import type { StoredAlert } from "@/context/AlertContext";

/**
 * Parses log file lines with format:
 * timestamp | ip | attack | prediction | risk_score | severity | payload
 */
export interface ParsedLogEntry {
  timestamp: string;
  ip: string;
  attack: string;
  prediction: string;
  risk_score: number;
  severity: "Critical" | "High" | "Medium" | "Low";
  payload: string;
}

const normalizeSeverity = (raw: string): StoredAlert["severity"] => {
  const s = raw.trim().toLowerCase();
  if (s === "critical") return "Critical";
  if (s === "high") return "High";
  if (s === "medium") return "Medium";
  return "Low";
};

export const parseLogLine = (line: string): ParsedLogEntry | null => {
  const parts = line.split("|").map((p) => p.trim());
  if (parts.length < 7) return null;

  const riskScore = parseInt(parts[4], 10);
  if (isNaN(riskScore)) return null;

  return {
    timestamp: parts[0],
    ip: parts[1],
    attack: parts[2],
    prediction: parts[3],
    risk_score: Math.max(0, Math.min(100, riskScore)),
    severity: normalizeSeverity(parts[5]),
    payload: parts.slice(6).join("|"),
  };
};

export const parseLogFile = (content: string): { entries: ParsedLogEntry[]; errors: number } => {
  const lines = content.split("\n").filter((l) => l.trim());
  let errors = 0;
  const entries: ParsedLogEntry[] = [];

  for (const line of lines) {
    const entry = parseLogLine(line);
    if (entry) {
      entries.push(entry);
    } else {
      errors++;
    }
  }

  return { entries, errors };
};

export const logEntriesToAlerts = (entries: ParsedLogEntry[]): Omit<StoredAlert, "id">[] => {
  return entries.map((e) => ({
    timestamp: e.timestamp,
    prediction: e.prediction,
    risk_score: e.risk_score,
    explanation: `Attack: ${e.attack} | Payload: ${e.payload}`,
    severity: e.severity,
    sourceIP: e.ip,
    status: e.risk_score >= 60 ? "Active" as const : "Investigating" as const,
    inputSnippet: e.payload.slice(0, 100),
  }));
};
