import { motion } from "framer-motion";
import { FileText, Table, Braces, Code, Terminal, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

const fileTypeIcons: Record<string, { icon: typeof FileText; color: string }> = {
  log: { icon: FileText, color: "#4361EE" },
  txt: { icon: FileText, color: "#4361EE" },
  csv: { icon: Table, color: "#00F5D4" },
  json: { icon: Braces, color: "#FFC107" },
  py: { icon: Code, color: "#A855F7" },
  xml: { icon: Terminal, color: "#FF8C00" },
  sh: { icon: Terminal, color: "#FF8C00" },
};

const recentFiles = [
  { name: "apache_access.log", date: "Dec 15, 2024", risk: "Critical", score: 94, ext: "log" },
  { name: "auth_failures.csv", date: "Dec 14, 2024", risk: "High", score: 78, ext: "csv" },
  { name: "network_dump.json", date: "Dec 14, 2024", risk: "Medium", score: 52, ext: "json" },
  { name: "scan_results.txt", date: "Dec 13, 2024", risk: "Low", score: 23, ext: "txt" },
  { name: "payload_check.py", date: "Dec 13, 2024", risk: "High", score: 81, ext: "py" },
];

const riskColors: Record<string, string> = {
  Critical: "#FF4560",
  High: "#FFC107",
  Medium: "#00F5D4",
  Low: "#4361EE",
};

const RecentUploads = () => (
  <div className="glass-card rounded-xl p-5">
    <h3 className="mb-3 text-sm font-semibold text-foreground">Recent Uploads</h3>
    <div className="space-y-2">
      {recentFiles.map((file, i) => {
        const cfg = fileTypeIcons[file.ext] || fileTypeIcons.txt;
        const Icon = cfg.icon;
        return (
          <motion.div
            key={file.name}
            className="flex items-center gap-3 rounded-lg bg-secondary/20 p-3 transition-colors hover:bg-secondary/40"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.35 }}
          >
            <div
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
              style={{ backgroundColor: `${cfg.color}15` }}
            >
              <Icon className="h-4 w-4" style={{ color: cfg.color }} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="truncate text-xs font-medium text-foreground">{file.name}</p>
              <p className="text-[10px] text-muted-foreground">{file.date}</p>
            </div>
            <span
              className="shrink-0 rounded-full px-2 py-0.5 text-[9px] font-bold"
              style={{ backgroundColor: `${riskColors[file.risk]}15`, color: riskColors[file.risk] }}
            >
              {file.risk}
            </span>
            <Button variant="ghost" size="sm" className="h-7 px-2 text-[10px] text-primary hover:bg-primary/10">
              <RotateCcw className="mr-1 h-3 w-3" /> Re-analyze
            </Button>
          </motion.div>
        );
      })}
    </div>
  </div>
);

export default RecentUploads;
