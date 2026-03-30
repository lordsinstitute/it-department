import { useAlerts } from "@/context/AlertContext";

const ThreatTicker = () => {
  const { alerts } = useAlerts();

  const recentAlerts = alerts.slice(0, 8);

  const items = recentAlerts.length > 0
    ? recentAlerts.map((a) => {
        const icon = a.severity === "Critical" ? "🔴" : a.severity === "High" ? "🟠" : a.severity === "Medium" ? "🟡" : "🟢";
        return `${icon} ${a.severity.toUpperCase()}: ${a.prediction} from ${a.sourceIP} — ${a.timestamp.slice(11, 16)}`;
      })
    : [
        "🟢 System active — No threats detected yet. Run an analysis to begin monitoring.",
      ];

  const text = items.join("  |  ") + "  |  ";

  return (
    <div className="overflow-hidden border-b border-border bg-background/90 py-1">
      <div className="flex whitespace-nowrap animate-[ticker_30s_linear_infinite]">
        <span className="font-mono text-[10px] sm:text-[11px] text-primary px-4">{text}</span>
        <span className="font-mono text-[10px] sm:text-[11px] text-primary px-4">{text}</span>
      </div>
      <style>{`
        @keyframes ticker {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `}</style>
    </div>
  );
};

export default ThreatTicker;
