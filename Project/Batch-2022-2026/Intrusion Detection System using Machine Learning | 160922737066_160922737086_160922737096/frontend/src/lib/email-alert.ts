// ── Storage keys ──
const ALERT_EMAIL_KEY = "ids_alert_email";
const EMAIL_SETTINGS_KEY = "ids_email_settings";
const EMAIL_HISTORY_KEY = "ids_email_history";
const RATE_LIMIT_KEY = "ids_email_rate_state";

// ── Types ──
export interface EmailSettings {
  enabled: boolean;
  severityFilters: {
    critical: boolean;
    high: boolean;
    medium: boolean;
    low: boolean;
  };
  maxPerHour: number;
  cooldownMinutes: number;
}

export interface EmailHistoryEntry {
  id: string;
  timestamp: string;
  prediction: string;
  severity: string;
  risk_score: number;
  email_sent_to: string;
  success: boolean;
  error?: string;
}

interface RateLimitState {
  sentTimestamps: number[];
}

// ── Default settings ──
const DEFAULT_SETTINGS: EmailSettings = {
  enabled: true,
  severityFilters: { critical: true, high: true, medium: false, low: false },
  maxPerHour: 10,
  cooldownMinutes: 5,
};

// ── Alert email ──
export const getAlertEmail = (): string => {
  return localStorage.getItem(ALERT_EMAIL_KEY) || "";
};

export const saveAlertEmail = (email: string) => {
  localStorage.setItem(ALERT_EMAIL_KEY, email);
};

// ── Settings ──
export const getEmailSettings = (): EmailSettings => {
  try {
    const raw = localStorage.getItem(EMAIL_SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
};

export const saveEmailSettings = (settings: EmailSettings) => {
  localStorage.setItem(EMAIL_SETTINGS_KEY, JSON.stringify(settings));
};

// ── History ──
export const getEmailHistory = (): EmailHistoryEntry[] => {
  try {
    const raw = localStorage.getItem(EMAIL_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

export const addEmailHistory = (entry: EmailHistoryEntry) => {
  const history = getEmailHistory();
  history.unshift(entry);
  localStorage.setItem(EMAIL_HISTORY_KEY, JSON.stringify(history.slice(0, 200)));
};

export const clearEmailHistory = () => {
  localStorage.setItem(EMAIL_HISTORY_KEY, JSON.stringify([]));
};

// ── Rate limiting ──
const getRateLimitState = (): RateLimitState => {
  try {
    const raw = localStorage.getItem(RATE_LIMIT_KEY);
    return raw ? JSON.parse(raw) : { sentTimestamps: [] };
  } catch {
    return { sentTimestamps: [] };
  }
};

const saveRateLimitState = (state: RateLimitState) => {
  localStorage.setItem(RATE_LIMIT_KEY, JSON.stringify(state));
};

export const checkRateLimit = (maxPerHour: number): boolean => {
  const state = getRateLimitState();
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  const recentSends = state.sentTimestamps.filter((t) => t > oneHourAgo);
  return recentSends.length < maxPerHour;
};

const recordSend = () => {
  const state = getRateLimitState();
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  state.sentTimestamps = [...state.sentTimestamps.filter((t) => t > oneHourAgo), Date.now()];
  saveRateLimitState(state);
};

export const getSendsInLastHour = (): number => {
  const state = getRateLimitState();
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  return state.sentTimestamps.filter((t) => t > oneHourAgo).length;
};

export const getLastSentTime = (): string | null => {
  const history = getEmailHistory();
  if (history.length === 0) return null;
  return history[0].timestamp;
};

// ── Severity helper ──
export const getSeverity = (riskScore: number): "low" | "medium" | "high" | "critical" => {
  if (riskScore >= 80) return "critical";
  if (riskScore >= 60) return "high";
  if (riskScore >= 30) return "medium";
  return "low";
};

// ── Should send email (checks all conditions) ──
export const shouldSendEmail = (riskScore: number): { allowed: boolean; reason?: string } => {
  const settings = getEmailSettings();
  const alertEmail = getAlertEmail();

  if (!settings.enabled) return { allowed: false, reason: "Email alerts are disabled" };
  if (!alertEmail) return { allowed: false, reason: "No recipient email set" };

  const severity = getSeverity(riskScore);
  if (!settings.severityFilters[severity])
    return { allowed: false, reason: `${severity.toUpperCase()} severity alerts are disabled` };

  if (!checkRateLimit(settings.maxPerHour))
    return { allowed: false, reason: `Rate limit reached (${settings.maxPerHour}/hour)` };

  return { allowed: true };
};

// ── Send email via FormSubmit ──
export const sendEmailAlert = async (data: {
  prediction: string;
  risk_score: number;
  explanation: string;
  timestamp: string;
  recipientEmail: string;
}): Promise<boolean> => {
  if (!data.recipientEmail) {
    console.warn("No recipient email — skipping alert");
    return false;
  }

  const severity = getSeverity(data.risk_score);
  const severityLabel = severity.toUpperCase();

  const payload = {
    prediction: data.prediction,
    risk_score: `${data.risk_score}`,
    severity: severityLabel,
    explanation: data.explanation,
    timestamp: data.timestamp,
    _subject: `🚨 IDS Alert — ${severityLabel} Threat Detected`,
  };

  console.log("📧 FormSubmit sending with params:", JSON.stringify(payload, null, 2));

  try {
    const response = await fetch(`https://formsubmit.co/ajax/${data.recipientEmail}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`FormSubmit responded with ${response.status}`);
    }

    // Record for rate limiting
    recordSend();

    // Save to history
    addEmailHistory({
      id: crypto.randomUUID(),
      timestamp: data.timestamp,
      prediction: data.prediction,
      severity: severityLabel,
      risk_score: data.risk_score,
      email_sent_to: data.recipientEmail,
      success: true,
    });

    return true;
  } catch (err: any) {
    addEmailHistory({
      id: crypto.randomUUID(),
      timestamp: data.timestamp,
      prediction: data.prediction,
      severity: severityLabel,
      risk_score: data.risk_score,
      email_sent_to: data.recipientEmail,
      success: false,
      error: err?.message || "Unknown error",
    });
    console.error("Email send failed:", err);
    return false;
  }
};
