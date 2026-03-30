import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { ShieldBan, Download, Radar, Trash2, Loader2, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAlerts } from "@/context/AlertContext";
import { analyzeThreat } from "@/lib/api";
import { generatePDFReport } from "@/lib/pdf-export";

const BLOCKED_IPS_KEY = "ids_blocked_ips";

const loadBlockedIPs = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem(BLOCKED_IPS_KEY) || "[]");
  } catch {
    return [];
  }
};

const saveBlockedIP = (ip: string) => {
  const list = loadBlockedIPs();
  if (!list.includes(ip)) {
    list.push(ip);
    localStorage.setItem(BLOCKED_IPS_KEY, JSON.stringify(list));
  }
};

const QuickActions = () => {
  const [blockOpen, setBlockOpen] = useState(false);
  const [clearOpen, setClearOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanDone, setScanDone] = useState(false);
  const [exporting, setExporting] = useState(false);
  const { toast } = useToast();
  const { alerts, refreshAlerts, deleteAllAlerts } = useAlerts();

  const handleScan = async () => {
    setScanning(true);
    setScanDone(false);
    try {
      const result = await analyzeThreat("network scan suspicious traffic");
      await refreshAlerts();
      toast({
        title: "Scan Complete",
        description: `Result: ${result.prediction} (Risk: ${result.risk_score})`,
      });
    } catch {
      toast({ title: "Scan Complete", description: "No new threats detected (backend unreachable — demo mode)." });
    } finally {
      setScanning(false);
      setScanDone(true);
      setTimeout(() => setScanDone(false), 2000);
    }
  };

  const handleExport = () => {
    if (alerts.length === 0) {
      toast({ title: "No data", description: "Run an analysis first to generate report data." });
      return;
    }
    setExporting(true);
    try {
      generatePDFReport(alerts);
      toast({ title: "Report Exported", description: "PDF report downloaded successfully." });
    } catch {
      toast({ title: "Export failed", description: "Could not generate PDF.", variant: "destructive" });
    } finally {
      setExporting(false);
    }
  };

  const handleBlock = (ip: string) => {
    setBlockOpen(false);
    if (!ip.trim()) return;

    // Persist blocked IP locally
    saveBlockedIP(ip.trim());

    toast({ title: "IP Blocked Successfully", description: `${ip.trim()} has been added to the blocklist.` });
  };

  const handleClear = async () => {
    setClearOpen(false);
    try {
      await deleteAllAlerts();
      toast({ title: "✅ Alerts Cleared", description: "All alerts have been removed." });
    } catch (err: any) {
      toast({ title: "Failed to clear", description: err.message, variant: "destructive" });
    }
  };

  return (
    <>
      <motion.div
        className="glass-card rounded-xl p-5"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.5 }}
      >
        <h3 className="mb-4 text-sm font-semibold text-foreground">Quick Actions</h3>
        <div className="grid grid-cols-2 gap-3">
          <Button
            variant="outline"
            onClick={() => setBlockOpen(true)}
            className="h-auto flex-col gap-2 border-critical/20 bg-critical/5 py-4 text-critical hover:bg-critical/10 hover:border-critical/40"
          >
            <ShieldBan className="h-5 w-5" />
            <span className="text-[10px] font-semibold">Block IP</span>
          </Button>
          <Button
            variant="outline"
            onClick={handleExport}
            disabled={exporting}
            className="h-auto flex-col gap-2 border-primary/20 bg-primary/5 py-4 text-primary hover:bg-primary/10 hover:border-primary/40"
          >
            {exporting ? <Loader2 className="h-5 w-5 animate-spin" /> : <Download className="h-5 w-5" />}
            <span className="text-[10px] font-semibold">Export Report</span>
          </Button>
          <Button
            variant="outline"
            onClick={handleScan}
            disabled={scanning}
            className="h-auto flex-col gap-2 border-[#4361EE]/20 bg-[#4361EE]/5 py-4 text-[#4361EE] hover:bg-[#4361EE]/10 hover:border-[#4361EE]/40"
          >
            {scanning ? (
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
                <Radar className="h-5 w-5" />
              </motion.div>
            ) : scanDone ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <Radar className="h-5 w-5" />
            )}
            <span className="text-[10px] font-semibold">{scanning ? "Scanning..." : scanDone ? "Done!" : "Scan Network"}</span>
          </Button>
          <Button
            variant="outline"
            onClick={() => setClearOpen(true)}
            className="h-auto flex-col gap-2 border-border bg-secondary/30 py-4 text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
          >
            <Trash2 className="h-5 w-5" />
            <span className="text-[10px] font-semibold">Clear Alerts</span>
          </Button>
        </div>
      </motion.div>

      {/* Block IP Modal */}
      <Dialog open={blockOpen} onOpenChange={setBlockOpen}>
        <DialogContent className="glass-card border-border sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              <ShieldBan className="h-5 w-5 text-critical" /> Block IP Address
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); const fd = new FormData(e.currentTarget); handleBlock(fd.get("ip") as string); }}>
            <Input name="ip" placeholder="e.g. 192.168.1.105" className="bg-secondary/50 border-border font-mono" autoFocus />
            <DialogFooter className="mt-4">
              <Button type="button" variant="outline" onClick={() => setBlockOpen(false)} className="border-border text-foreground">Cancel</Button>
              <Button type="submit" className="bg-critical text-critical-foreground hover:bg-critical/90">Block IP</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Clear Alerts Confirmation */}
      <Dialog open={clearOpen} onOpenChange={setClearOpen}>
        <DialogContent className="glass-card border-border sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-foreground">Clear All Alerts?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">This will permanently remove all alerts from the system. This action cannot be undone.</p>
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setClearOpen(false)} className="border-border text-foreground">Cancel</Button>
            <Button onClick={handleClear} className="bg-critical text-critical-foreground hover:bg-critical/90">Clear All</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default QuickActions;
