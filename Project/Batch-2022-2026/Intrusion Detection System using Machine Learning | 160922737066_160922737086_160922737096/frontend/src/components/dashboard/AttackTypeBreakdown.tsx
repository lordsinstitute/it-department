import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import ChartHelpTooltip from "@/components/dashboard/ChartHelpTooltip";
import { useAlerts } from "@/context/AlertContext";

const COLORS: Record<string, string> = {
  "DoS/DDoS": "#FF4560",
  "Port Scan": "#FF8C00",
  "SQL Injection": "#FFC107",
  "Brute Force": "#A855F7",
  "XSS Attack": "#4361EE",
  "Malicious": "#FF4560",
  "Normal": "#22C55E",
  "Other": "#6B7280",
};

const categorize = (prediction: string): string => {
  const p = prediction.toLowerCase();
  if (p.includes("dos") || p.includes("ddos") || p.includes("neptune") || p.includes("smurf")) return "DoS/DDoS";
  if (p.includes("scan") || p.includes("portsweep") || p.includes("ipsweep") || p.includes("nmap")) return "Port Scan";
  if (p.includes("sql") || p.includes("injection")) return "SQL Injection";
  if (p.includes("brute") || p.includes("guess") || p.includes("login")) return "Brute Force";
  if (p.includes("xss") || p.includes("script")) return "XSS Attack";
  if (p.includes("normal") || p.includes("benign")) return "Normal";
  return "Malicious";
};

const AttackTypeBreakdown = () => {
  const { alerts } = useAlerts();

  const counts: Record<string, number> = {};
  alerts.forEach((a) => {
    const cat = categorize(a.prediction);
    counts[cat] = (counts[cat] || 0) + 1;
  });

  const data = Object.entries(counts)
    .map(([type, count]) => ({ type, count, color: COLORS[type] || "#6B7280" }))
    .sort((a, b) => b.count - a.count);

  if (data.length === 0) {
    return (
      <motion.div
        className="glass-card rounded-xl p-5"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.5 }}
      >
        <div className="mb-4 flex items-center gap-2">
          <h3 className="text-sm font-semibold text-foreground">Attack Type Breakdown</h3>
          <ChartHelpTooltip text="Distribution of detected attacks categorized by attack vector type." />
        </div>
        <div className="flex items-center justify-center h-[220px]">
          <p className="text-xs text-muted-foreground">No data yet — run an analysis first</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="glass-card rounded-xl p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7, duration: 0.5 }}
    >
      <div className="mb-4 flex items-center gap-2">
        <h3 className="text-sm font-semibold text-foreground">Attack Type Breakdown</h3>
        <ChartHelpTooltip text="Distribution of detected attacks categorized by attack vector type — from real analyses." />
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical">
          <XAxis type="number" tick={{ fill: "hsl(215, 20%, 50%)", fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="type" tick={{ fill: "hsl(215, 20%, 50%)", fontSize: 10 }} axisLine={false} tickLine={false} width={85} />
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <div className="glass-card rounded-lg p-2.5 shadow-xl">
                  <p className="font-mono text-xs text-foreground">{payload[0].payload.type}: <strong>{payload[0].value}</strong></p>
                </div>
              ) : null
            }
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={16} isAnimationActive animationDuration={1200} animationEasing="ease-out">
            {data.map((d, i) => <Cell key={i} fill={d.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
};

export default AttackTypeBreakdown;
