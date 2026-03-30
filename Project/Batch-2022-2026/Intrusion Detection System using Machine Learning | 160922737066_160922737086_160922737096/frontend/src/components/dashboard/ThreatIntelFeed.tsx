import { motion } from "framer-motion";
import { ExternalLink } from "lucide-react";
import ChartHelpTooltip from "@/components/dashboard/ChartHelpTooltip";

const feedItems = [
  { type: "CVE", name: "CVE-2024-1234 — Apache RCE", severity: "#FF4560", time: "2 hours ago", badgeColor: "#FF4560" },
  { type: "EXPLOIT", name: "Log4Shell Variant Detected in Wild", severity: "#FF8C00", time: "4 hours ago", badgeColor: "#FF8C00" },
  { type: "MALWARE", name: "BlackCat Ransomware New Payload", severity: "#FF4560", time: "6 hours ago", badgeColor: "#A855F7" },
  { type: "PHISHING", name: "O365 Credential Harvesting Campaign", severity: "#FFC107", time: "8 hours ago", badgeColor: "#4361EE" },
  { type: "ZERO-DAY", name: "Chrome V8 Engine Memory Corruption", severity: "#FF4560", time: "12 hours ago", badgeColor: "#FF4560" },
];

const ThreatIntelFeed = () => (
  <motion.div
    className="glass-card rounded-xl p-5"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.9, duration: 0.5 }}
  >
    <div className="mb-4 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-foreground">Threat Intelligence Feed</h3>
        <ChartHelpTooltip text="Latest threat advisories from global cybersecurity intelligence sources." />
      </div>
      <button className="flex items-center gap-1 text-[10px] font-medium text-primary hover:underline">
        View all <ExternalLink className="h-3 w-3" />
      </button>
    </div>
    <div className="space-y-2">
      {feedItems.map((item, i) => (
        <motion.div
          key={i}
          className="group flex items-center gap-3 rounded-lg border border-transparent bg-secondary/20 p-3 transition-all hover:border-primary/30 hover:glow-primary cursor-pointer"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.0 + i * 0.08, duration: 0.4 }}
          style={{ ["--tw-shadow" as string]: "var(--glow-primary)" }}
          whileHover={{ boxShadow: "0 0 15px hsl(168 100% 48% / 0.15)" }}
        >
          <span
            className="shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold"
            style={{ backgroundColor: `${item.badgeColor}20`, color: item.badgeColor }}
          >
            {item.type}
          </span>
          <div className="flex-1 min-w-0">
            <p className="truncate text-xs font-medium text-foreground">{item.name}</p>
          </div>
          <div className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: item.severity }} />
          <span className="shrink-0 text-[10px] text-muted-foreground">{item.time}</span>
          <button className="shrink-0 text-[10px] font-medium text-primary opacity-0 group-hover:opacity-100 transition-opacity">
            Details
          </button>
        </motion.div>
      ))}
    </div>
  </motion.div>
);

export default ThreatIntelFeed;
