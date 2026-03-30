import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Search, Download, FileDown, Loader2, Eye, FileText, Clock, Trash2,
  AlertTriangle, Shield, Wifi, Brain, UploadCloud,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAlerts, type StoredAlert } from "@/context/AlertContext";
import { useAuth } from "@/context/AuthContext";
import { generatePDFReport } from "@/lib/pdf-export";
import { parseLogFile, logEntriesToAlerts } from "@/lib/log-parser";

const sevColor: Record<StoredAlert["severity"], string> = {
  Critical: "bg-critical/15 text-critical",
  High: "bg-high/15 text-high",
  Medium: "bg-warning/15 text-warning",
  Low: "bg-primary/15 text-primary",
};
const sevBarColor: Record<StoredAlert["severity"], string> = {
  Critical: "bg-critical",
  High: "bg-high",
  Medium: "bg-warning",
  Low: "bg-primary",
};
const statusColor: Record<StoredAlert["status"], string> = {
  Active: "bg-critical/15 text-critical",
  Investigating: "bg-warning/15 text-warning",
  Resolved: "bg-success/15 text-success",
};

const AlertLog = () => {
  const [search, setSearch] = useState("");
  const [sevFilter, setSevFilter] = useState<"all" | StoredAlert["severity"]>("all");
  const [pdfLoading, setPdfLoading] = useState(false);
  const [viewAlert, setViewAlert] = useState<StoredAlert | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  const { alerts, totalAlerts, criticalCount, resolvedCount, activeCount, loading, error, deleteAlert, deleteAllAlerts } = useAlerts();
  const { isAdmin } = useAuth();

  // Log upload is disabled in API mode — alerts come from backend


  const filtered = alerts.filter((a) => {
    const q = search.toLowerCase();
    const matchSearch = !q || a.id.toLowerCase().includes(q) || a.prediction.toLowerCase().includes(q) || a.sourceIP.includes(q);
    const matchSev = sevFilter === "all" || a.severity === sevFilter;
    return matchSearch && matchSev;
  });

  const exportCSV = () => {
    if (filtered.length === 0) {
      toast({ title: "No data", description: "No alerts to export." });
      return;
    }
    const headers = "ID,Timestamp,Prediction,Severity,Risk Score,Source IP,Status\n";
    const rows = filtered.map((a) => `${a.id},${a.timestamp},${a.prediction},${a.severity},${a.risk_score},${a.sourceIP},${a.status}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ids_alert_log_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    toast({ title: "CSV Exported", description: `${filtered.length} alerts exported.` });
  };

  const handlePdf = () => {
    if (filtered.length === 0) {
      toast({ title: "No data", description: "No alerts to export." });
      return;
    }
    setPdfLoading(true);
    try {
      generatePDFReport(filtered);
      toast({ title: "Report downloaded", description: "PDF report saved successfully." });
    } catch {
      toast({ title: "Export failed", variant: "destructive" });
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-bold text-foreground">Alert History</h2>
          {isAdmin && alerts.length > 0 && (
            <Button
              variant="destructive"
              size="sm"
              className="text-xs"
              onClick={async () => {
                if (!confirm("Delete ALL alerts? This cannot be undone.")) return;
                await deleteAllAlerts();
                toast({ title: "All alerts deleted" });
              }}
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" /> Delete All
            </Button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="relative flex-1 min-w-[140px]">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search alerts..." className="w-full pl-9 bg-secondary/50 border-border text-xs" />
          </div>
          <Select value={sevFilter} onValueChange={(v) => setSevFilter(v as any)}>
            <SelectTrigger className="w-28 sm:w-32 bg-secondary/50 border-border text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="Critical">Critical</SelectItem>
              <SelectItem value="High">High</SelectItem>
              <SelectItem value="Medium">Medium</SelectItem>
              <SelectItem value="Low">Low</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={exportCSV} className="border-primary/40 text-primary hover:bg-primary/10 text-xs">
            <Download className="mr-1.5 h-3.5 w-3.5" /> <span className="hidden sm:inline">Export</span> CSV
          </Button>
          <Button size="sm" onClick={handlePdf} disabled={pdfLoading} className="bg-primary text-primary-foreground text-xs">
            {pdfLoading ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <FileDown className="mr-1.5 h-3.5 w-3.5" />}
            {pdfLoading ? "..." : <span className="hidden sm:inline">Download</span>} PDF
          </Button>
        </div>
      </div>

      {/* Loading / Error */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <span className="ml-2 text-sm text-muted-foreground">Loading alerts from backend...</span>
        </div>
      )}
      {error && (
        <div className="glass-card rounded-xl p-4 border border-critical/30 text-center">
          <p className="text-xs text-critical">Backend not reachable: {error}</p>
        </div>
      )}

      {/* Stats bar */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Total", value: totalAlerts, color: "text-primary" },
          { label: "Critical", value: criticalCount, color: "text-critical" },
          { label: "Resolved", value: resolvedCount, color: "text-success" },
          { label: "Active", value: activeCount, color: "text-warning" },
        ].map((s) => (
          <div key={s.label} className="glass-card rounded-xl px-4 py-3 text-center">
            <p className={`font-mono text-xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <div className="glass-card rounded-xl p-10 text-center">
          <p className="text-sm text-muted-foreground">
            {totalAlerts === 0
              ? "No alerts yet — run an analysis from the Analyze page to generate alerts."
              : "No alerts match your filters."}
          </p>
        </div>
      ) : (
        <>
          {/* Mobile card view */}
          <div className="space-y-3 lg:hidden">
            <AnimatePresence>
              {filtered.map((a, i) => (
                <motion.div
                  key={a.id}
                  className="glass-card rounded-xl p-3 sm:p-4 space-y-2"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ delay: i * 0.03 }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[10px] font-medium text-primary truncate">{a.id}</span>
                    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold ${sevColor[a.severity]}`}>
                      {a.severity}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-foreground">{a.prediction}</span>
                    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium ${statusColor[a.status]}`}>
                      {a.status === "Active" && <span className="h-1.5 w-1.5 rounded-full bg-critical animate-pulse" />}
                      {a.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 flex-1">
                      <div className="h-1.5 flex-1 max-w-[100px] overflow-hidden rounded-full bg-secondary">
                        <motion.div
                          className={`h-full rounded-full ${sevBarColor[a.severity]}`}
                          initial={{ width: 0 }}
                          animate={{ width: `${a.risk_score}%` }}
                          transition={{ duration: 0.8 }}
                        />
                      </div>
                      <span className="font-mono text-xs text-muted-foreground">{a.risk_score}</span>
                    </div>
                    <span className="font-mono text-[10px] text-muted-foreground">{a.sourceIP}</span>
                  </div>
                  <div className="flex items-center justify-between pt-1 border-t border-border">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <span className="font-mono text-[10px] text-muted-foreground">{a.timestamp}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <button onClick={() => setViewAlert(a)} className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
                        <Eye className="h-3.5 w-3.5" />
                      </button>
                      <button onClick={() => { generatePDFReport([a]); toast({ title: "PDF Downloaded", description: `Report for ${a.id}` }); }} className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
                        <FileText className="h-3.5 w-3.5" />
                      </button>
                      {isAdmin && (
                        <button onClick={async () => { if (!confirm(`Delete alert ${a.id}?`)) return; try { await deleteAlert(a.id); toast({ title: "Alert deleted", description: a.id }); } catch (err: any) { toast({ title: "Delete failed", description: err.message, variant: "destructive" }); } }} className="rounded p-1 text-muted-foreground hover:bg-destructive/20 hover:text-destructive transition-colors">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* Desktop table view */}
          <div className="glass-card overflow-hidden rounded-xl hidden lg:block">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    {["ID", "Timestamp", "Prediction", "Severity", "Risk Score", "Source IP", "Status", "Actions"].map((h) => (
                      <th key={h} className="px-4 py-3 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  <AnimatePresence>
                    {filtered.map((a, i) => (
                      <motion.tr
                        key={a.id}
                        className="transition-colors hover:bg-secondary/30"
                        initial={{ opacity: 0, x: -15 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 15 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                      >
                        <td className="px-4 py-3">
                          <span className="font-mono text-xs font-medium text-primary">{a.id}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1.5">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span className="font-mono text-xs text-muted-foreground">{a.timestamp}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs font-medium text-foreground">{a.prediction}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex rounded-full px-2.5 py-0.5 text-[10px] font-bold ${sevColor[a.severity]}`}>
                            {a.severity}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
                              <motion.div
                                className={`h-full rounded-full ${sevBarColor[a.severity]}`}
                                initial={{ width: 0 }}
                                animate={{ width: `${a.risk_score}%` }}
                                transition={{ delay: 0.2 + i * 0.04, duration: 1 }}
                              />
                            </div>
                            <span className="font-mono text-xs text-muted-foreground">{a.risk_score}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="font-mono text-xs text-muted-foreground">{a.sourceIP}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-medium ${statusColor[a.status]}`}>
                            {a.status === "Active" && <span className="h-1.5 w-1.5 rounded-full bg-critical animate-pulse" />}
                            {a.status}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1">
                            <button onClick={() => setViewAlert(a)} className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors" title="View Report">
                              <Eye className="h-3.5 w-3.5" />
                            </button>
                            <button onClick={() => { generatePDFReport([a]); toast({ title: "PDF Downloaded", description: `Report for ${a.id}` }); }} className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors" title="Download PDF">
                              <FileText className="h-3.5 w-3.5" />
                            </button>
                            {isAdmin && (
                              <button onClick={async () => { if (!confirm(`Delete alert ${a.id}?`)) return; try { await deleteAlert(a.id); toast({ title: "Alert deleted", description: a.id }); } catch (err: any) { toast({ title: "Delete failed", description: err.message, variant: "destructive" }); } }} className="rounded p-1 text-muted-foreground hover:bg-destructive/20 hover:text-destructive transition-colors" title="Delete Alert">
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            )}
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Alert Detail Report Dialog */}
      <Dialog open={!!viewAlert} onOpenChange={(open) => !open && setViewAlert(null)}>
        <DialogContent className="max-w-lg border-border bg-card text-foreground">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-lg">
              <Shield className="h-5 w-5 text-primary" />
              Threat Report — {viewAlert?.id}
            </DialogTitle>
          </DialogHeader>
          {viewAlert && (
            <div className="space-y-4">
              {/* Severity + Score header */}
              <div className="flex items-center justify-between rounded-lg bg-secondary/40 p-4">
                <div className="flex items-center gap-3">
                  <AlertTriangle className={`h-8 w-8 ${viewAlert.severity === "Critical" ? "text-critical" : viewAlert.severity === "High" ? "text-high" : viewAlert.severity === "Medium" ? "text-warning" : "text-primary"}`} />
                  <div>
                    <p className={`text-xl font-bold ${viewAlert.severity === "Critical" ? "text-critical" : viewAlert.severity === "High" ? "text-high" : viewAlert.severity === "Medium" ? "text-warning" : "text-primary"}`}>
                      {viewAlert.severity}
                    </p>
                    <p className="text-xs text-muted-foreground">Threat Level</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold text-foreground">{viewAlert.risk_score}<span className="text-sm text-muted-foreground">/100</span></p>
                  <p className="text-xs text-muted-foreground">Risk Score</p>
                </div>
              </div>

              {/* Info grid */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Prediction", value: viewAlert.prediction, icon: Brain },
                  { label: "Status", value: viewAlert.status, icon: Shield },
                  { label: "Source IP", value: viewAlert.sourceIP, icon: Wifi },
                  { label: "Timestamp", value: viewAlert.timestamp, icon: Clock },
                ].map((item) => (
                  <div key={item.label} className="rounded-lg bg-secondary/30 p-3">
                    <div className="flex items-center gap-1.5 mb-1">
                      <item.icon className="h-3 w-3 text-muted-foreground" />
                      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{item.label}</span>
                    </div>
                    <p className="text-sm font-medium text-foreground">{item.value}</p>
                  </div>
                ))}
              </div>

              {/* AI Explanation */}
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-primary">AI Reasoning</p>
                <p className="text-sm leading-relaxed text-muted-foreground">{viewAlert.explanation}</p>
              </div>

              {/* Input snippet */}
              {viewAlert.inputSnippet && (
                <div className="rounded-lg bg-secondary/30 p-3">
                  <p className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">Input Snippet</p>
                  <code className="block text-xs font-mono text-muted-foreground whitespace-pre-wrap">{viewAlert.inputSnippet}</code>
                </div>
              )}

              {/* Download button */}
              <Button
                className="w-full"
                variant="outline"
                onClick={() => {
                  generatePDFReport([viewAlert]);
                  toast({ title: "PDF Downloaded", description: `Report for ${viewAlert.id}` });
                }}
              >
                <FileDown className="mr-2 h-4 w-4" />
                Download PDF Report
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AlertLog;
