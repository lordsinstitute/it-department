import { motion } from "framer-motion";
import { useAlerts } from "@/context/AlertContext";
import { Clock, ExternalLink } from "lucide-react";
import ChartHelpTooltip from "@/components/dashboard/ChartHelpTooltip";

const severityColors: Record<string, string> = {
  Critical: "#FF4560",
  High: "#FFC107",
  Medium: "#00F5D4",
  Low: "#4361EE",
};

const statusStyles: Record<string, string> = {
  Active: "bg-critical/10 text-critical",
  Investigating: "bg-warning/10 text-warning",
  Resolved: "bg-success/10 text-success",
};

const AlertsTable = () => {
  const { alerts } = useAlerts();
  const recent = alerts.slice(0, 6);

  return (
    <div className="glass-card rounded-xl">
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-foreground">Recent Threat Feed</h3>
            <ChartHelpTooltip text="Most recent predictions from backend API with severity scoring." />
          </div>
          <p className="text-xs text-muted-foreground">
            {recent.length > 0 ? "Latest predictions from backend" : "No predictions yet"}
          </p>
        </div>
      </div>

      {recent.length === 0 ? (
        <div className="p-8 text-center">
          <p className="text-xs text-muted-foreground">No data yet — run an analysis to see results here.</p>
        </div>
      ) : (
        <div className="overflow-x-auto -mx-1">
          <table className="w-full min-w-[640px]">
            <thead>
              <tr className="border-b border-border">
                {["Alert ID", "Timestamp", "Prediction", "Severity", "Risk Score", "Source IP", "Status"].map((h) => (
                  <th key={h} className="px-3 sm:px-4 py-2.5 sm:py-3 text-left text-[9px] sm:text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {recent.map((alert, i) => (
                <motion.tr
                  key={alert.id}
                  className="transition-colors hover:bg-secondary/30"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08, duration: 0.4, ease: "easeOut" }}
                >
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs font-medium text-primary">{alert.id}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <span className="font-mono text-xs text-muted-foreground">{alert.timestamp}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs font-medium text-foreground">{alert.prediction}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: severityColors[alert.severity] }} />
                      <span className="text-xs font-semibold" style={{ color: severityColors[alert.severity] }}>{alert.severity}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ backgroundColor: severityColors[alert.severity] }}
                          initial={{ width: 0 }}
                          animate={{ width: `${alert.risk_score}%` }}
                          transition={{ delay: 0.2 + i * 0.08, duration: 1.2 }}
                        />
                      </div>
                      <span className="font-mono text-xs text-muted-foreground">{alert.risk_score}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-muted-foreground">{alert.sourceIP}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${statusStyles[alert.status]}`}>
                      {alert.status === "Active" && <span className="h-1.5 w-1.5 rounded-full bg-critical animate-pulse" />}
                      {alert.status}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AlertsTable;
