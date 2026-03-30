import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuth } from "@/context/AuthContext";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Mail, Send, Eye, Save, Loader2, Trash2,
  CheckCircle, XCircle, Clock, Search, Bell, BellOff, Filter,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  getAlertEmail, saveAlertEmail,
  sendEmailAlert, getEmailSettings, saveEmailSettings,
  getEmailHistory, clearEmailHistory, getSendsInLastHour, getLastSentTime,
  type EmailSettings, type EmailHistoryEntry,
} from "@/lib/email-alert";

const Notifications = () => {
  const { toast } = useToast();
  const { user } = useAuth();
  const [testLoading, setTestLoading] = useState(false);

  // Recipient email — default to logged-in user's email if none saved
  const savedEmail = getAlertEmail();
  const [alertEmail, setAlertEmail] = useState(savedEmail || user?.email || "");

  // Auto-fill from auth if no email is saved yet
  useEffect(() => {
    if (!savedEmail && user?.email) {
      setAlertEmail(user.email);
      saveAlertEmail(user.email);
    }
  }, [user?.email, savedEmail]);

  // Settings
  const existingSettings = getEmailSettings();
  const [enabled, setEnabled] = useState(existingSettings.enabled);
  const [severityFilters, setSeverityFilters] = useState(existingSettings.severityFilters);
  const [maxPerHour, setMaxPerHour] = useState(String(existingSettings.maxPerHour));
  const [cooldownMinutes, setCooldownMinutes] = useState(String(existingSettings.cooldownMinutes));

  // History
  const [history, setHistory] = useState<EmailHistoryEntry[]>(getEmailHistory());
  const [historySearch, setHistorySearch] = useState("");
  const [historySeverity, setHistorySeverity] = useState("all");

  const filteredHistory = useMemo(() => {
    return history.filter((h) => {
      if (historySeverity !== "all" && h.severity !== historySeverity.toUpperCase()) return false;
      if (historySearch && !h.email_sent_to.toLowerCase().includes(historySearch.toLowerCase()) && !h.prediction.toLowerCase().includes(historySearch.toLowerCase())) return false;
      return true;
    });
  }, [history, historySearch, historySeverity]);

  const sendsThisHour = getSendsInLastHour();
  const lastSent = getLastSentTime();

  const handleSave = () => {
    saveAlertEmail(alertEmail);
    const settings: EmailSettings = {
      enabled,
      severityFilters,
      maxPerHour: parseInt(maxPerHour) || 10,
      cooldownMinutes: parseInt(cooldownMinutes) || 5,
    };
    saveEmailSettings(settings);
    toast({ title: "✅ Settings saved", description: "All notification settings updated." });
  };

  const handleTest = async () => {
    if (!alertEmail) {
      toast({ title: "⚠ No email set", description: "Please enter a recipient email first.", variant: "destructive" });
      return;
    }
    setTestLoading(true);
    try {
      const sent = await sendEmailAlert({
        prediction: "SQL Injection (Test Alert)",
        risk_score: 95,
        explanation: "This is a test alert from IDS Shield to verify email configuration.",
        timestamp: new Date().toISOString().replace("T", " ").slice(0, 19),
        recipientEmail: alertEmail,
      });
      setHistory(getEmailHistory());
      if (sent) {
        toast({ title: "✅ Test alert sent", description: `Sent to ${alertEmail}` });
      } else {
        toast({ title: "⚠ Send failed", description: "Check console for details.", variant: "destructive" });
      }
    } catch {
      toast({ title: "❌ Send failed", description: "FormSubmit request failed.", variant: "destructive" });
    } finally {
      setTestLoading(false);
    }
  };

  const handleClearHistory = () => {
    clearEmailHistory();
    setHistory([]);
    toast({ title: "History cleared" });
  };

  const notifRules = [
    { key: "critical" as const, label: "Critical Threats (80–100)", dot: "bg-critical", desc: "Immediate alert on critical threats" },
    { key: "high" as const, label: "High Threats (60–80)", dot: "bg-high", desc: "Alert on high-risk detections" },
    { key: "medium" as const, label: "Medium Threats (30–60)", dot: "bg-warning", desc: "Alert on medium-risk events" },
    { key: "low" as const, label: "Low Threats (0–30)", dot: "bg-success", desc: "Alert on low-risk events" },
  ];

  const inputClass = "bg-secondary/50 border-border focus:border-primary focus:shadow-[0_0_12px_rgba(0,245,212,0.15)] text-foreground";

  return (
    <div className="space-y-6">
      {/* Header with status */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-foreground">Email Alert Configuration</h2>
          <p className="text-xs text-muted-foreground">Configure automated threat notifications via FormSubmit</p>
        </div>
        <div className="flex flex-wrap items-center gap-3 sm:gap-4 text-xs">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Clock className="h-3.5 w-3.5" />
            {lastSent ? `Last sent: ${lastSent}` : "No emails sent yet"}
          </span>
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Mail className="h-3.5 w-3.5" /> {sendsThisHour}/{maxPerHour} this hour
          </span>
          <span className={`flex items-center gap-1.5 font-semibold ${enabled ? "text-primary" : "text-muted-foreground"}`}>
            {enabled ? <Bell className="h-3.5 w-3.5" /> : <BellOff className="h-3.5 w-3.5" />}
            {enabled ? "Alerts Active" : "Alerts Disabled"}
          </span>
        </div>
      </div>

      {/* Master toggle */}
      <div className="glass-card rounded-xl p-4 flex items-center justify-between border border-primary/20">
        <div className="flex items-center gap-3">
          <Bell className="h-5 w-5 text-primary" />
          <div>
            <p className="text-sm font-semibold text-foreground">Enable Email Alerts</p>
            <p className="text-[10px] text-muted-foreground">Master switch — when off, no emails will be sent automatically</p>
          </div>
        </div>
        <Switch checked={enabled} onCheckedChange={setEnabled} />
      </div>

      {/* FormSubmit Configuration */}
      <div className="glass-card rounded-xl p-6 space-y-5 border border-primary/20">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Mail className="h-4 w-4 text-primary" /> FormSubmit Configuration
        </h3>
        <p className="text-[10px] text-muted-foreground">
          Emails are sent via{" "}
          <a href="https://formsubmit.co/" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">FormSubmit.co</a>
          {" "}— no API keys needed. Just enter your email below. You may need to confirm your email on first use.
        </p>
        <div className="space-y-1.5">
          <Label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Alert Recipient Email</Label>
          <Input value={alertEmail} onChange={(e) => setAlertEmail(e.target.value)} placeholder="your-email@gmail.com" className={inputClass} />
        </div>
      </div>

      {/* Two column: Severity Filters + Rate Limiting */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Severity Filters */}
        <div className="glass-card rounded-xl p-6 space-y-5">
          <h3 className="text-sm font-semibold text-foreground">Severity Filters</h3>
          <p className="text-[10px] text-muted-foreground">Only send emails for selected severity levels</p>
          <div className="space-y-3">
            {notifRules.map((rule) => (
              <div key={rule.key} className="flex items-center justify-between rounded-lg bg-secondary/30 px-4 py-3">
                <div className="flex items-center gap-3">
                  <span className={`h-2.5 w-2.5 rounded-full ${rule.dot}`} />
                  <div>
                    <p className="text-xs font-medium text-foreground">{rule.label}</p>
                    <p className="text-[10px] text-muted-foreground">{rule.desc}</p>
                  </div>
                </div>
                <Switch
                  checked={severityFilters[rule.key]}
                  onCheckedChange={(v) => setSeverityFilters({ ...severityFilters, [rule.key]: v })}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Rate Limiting */}
        <div className="glass-card rounded-xl p-6 space-y-5">
          <h3 className="text-sm font-semibold text-foreground">Rate Limiting</h3>
          <p className="text-[10px] text-muted-foreground">Prevent email spam with configurable limits</p>
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label className="text-[10px] text-muted-foreground">Max emails per hour</Label>
              <Input type="number" value={maxPerHour} onChange={(e) => setMaxPerHour(e.target.value)} className={`${inputClass} w-32`} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[10px] text-muted-foreground">Cooldown between same attack type</Label>
              <Select value={cooldownMinutes} onValueChange={setCooldownMinutes}>
                <SelectTrigger className={inputClass}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 minute</SelectItem>
                  <SelectItem value="5">5 minutes</SelectItem>
                  <SelectItem value="15">15 minutes</SelectItem>
                  <SelectItem value="30">30 minutes</SelectItem>
                  <SelectItem value="60">1 hour</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="rounded-lg bg-secondary/30 px-4 py-3">
              <p className="text-xs text-foreground font-medium">Current Usage</p>
              <p className="text-[10px] text-muted-foreground mt-1">
                {sendsThisHour} / {maxPerHour} emails sent in the last hour
              </p>
              <div className="mt-2 h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${Math.min(100, (sendsThisHour / (parseInt(maxPerHour) || 10)) * 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Save + Test */}
      <div className="flex flex-wrap gap-3">
        <Button onClick={handleSave} className="bg-primary text-primary-foreground hover:bg-primary/90">
          <Save className="mr-2 h-4 w-4" /> Save
        </Button>
        <Button variant="outline" onClick={handleTest} disabled={testLoading} className="border-primary/40 text-primary hover:bg-primary/10">
          {testLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
          {testLoading ? "Sending..." : "Test Alert"}
        </Button>
      </div>

      {/* Email Alert History */}
      <div className="glass-card rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Mail className="h-4 w-4 text-primary" /> Email Alert History
            <span className="text-[10px] font-normal text-muted-foreground ml-1">({history.length} total)</span>
          </h3>
          <Button variant="ghost" size="sm" onClick={handleClearHistory} className="text-muted-foreground hover:text-destructive text-xs">
            <Trash2 className="mr-1 h-3 w-3" /> Clear
          </Button>
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              value={historySearch}
              onChange={(e) => setHistorySearch(e.target.value)}
              placeholder="Search by email or prediction..."
              className={`${inputClass} pl-9 h-8 text-xs`}
            />
          </div>
          <Select value={historySeverity} onValueChange={setHistorySeverity}>
            <SelectTrigger className={`${inputClass} w-36 h-8 text-xs`}>
              <Filter className="mr-1 h-3 w-3" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severities</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {filteredHistory.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-xs">
            <Mail className="h-8 w-8 mx-auto mb-2 opacity-30" />
            <p>No email alerts sent yet</p>
            <p className="text-[10px] mt-1">Alerts will appear here when triggered by analysis</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {filteredHistory.map((entry) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between rounded-lg bg-secondary/30 px-4 py-3"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {entry.success ? (
                    <CheckCircle className="h-4 w-4 text-primary shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 text-destructive shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-foreground truncate">{entry.prediction}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {entry.email_sent_to} · {entry.timestamp}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    entry.severity === "CRITICAL" ? "bg-critical/15 text-critical" :
                    entry.severity === "HIGH" ? "bg-high/15 text-high" :
                    entry.severity === "MEDIUM" ? "bg-warning/15 text-warning" :
                    "bg-success/15 text-success"
                  }`}>
                    {entry.severity}
                  </span>
                  <span className="text-[10px] font-mono text-muted-foreground">{entry.risk_score}</span>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Email Preview */}
      <div className="glass-card rounded-xl p-6 space-y-4">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Eye className="h-4 w-4 text-primary" /> Preview Alert Email
        </h3>
        <div className="rounded-xl border border-border bg-secondary/20 overflow-hidden">
          <div className="border-b border-border px-5 py-3 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground w-14">Subject:</span>
              <span className="font-semibold text-foreground">🚨 IDS Alert — CRITICAL Threat Detected</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground w-14">To:</span>
              <span className="text-foreground/80">{alertEmail || "recipient@example.com"}</span>
            </div>
          </div>
          <div className="p-5 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2">
              {[
                ["Prediction", "SQL Injection"],
                ["Risk Score", "98/100"],
                ["Severity", "CRITICAL"],
                ["Time", new Date().toISOString().replace("T", " ").slice(0, 19)],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center justify-between py-1 border-b border-border/50">
                  <span className="text-[10px] text-muted-foreground">{label}</span>
                  <span className="font-mono text-xs font-semibold text-foreground">{value}</span>
                </div>
              ))}
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">AI Reasoning:</p>
              <p className="text-xs text-foreground/80">Malicious SQL patterns detected in query parameters targeting database extraction.</p>
            </div>
            <p className="text-center text-[10px] text-muted-foreground pt-2 border-t border-border/50">
              Sent by IDS Shield — Adaptive Intrusion Detection System
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Notifications;
