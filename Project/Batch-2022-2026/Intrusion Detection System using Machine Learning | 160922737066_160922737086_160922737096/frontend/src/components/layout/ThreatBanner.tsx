import { AlertTriangle, X } from "lucide-react";
import { useState } from "react";
import { useAlerts } from "@/context/AlertContext";

const ThreatBanner = () => {
  const [visible, setVisible] = useState(true);
  const { criticalCount } = useAlerts();

  if (!visible || criticalCount === 0) return null;

  return (
    <div className="flex items-center justify-between bg-critical/10 border-b border-critical/20 px-3 sm:px-6 py-2">
      <div className="flex items-center gap-2 sm:gap-3 min-w-0">
        <AlertTriangle className="h-4 w-4 text-critical animate-threat-pulse shrink-0" />
        <span className="text-[10px] sm:text-xs font-semibold text-critical truncate">
          CRITICAL THREAT DETECTED
        </span>
        <span className="text-[10px] sm:text-xs text-critical/80 hidden sm:inline">
          — {criticalCount} unresolved alert{criticalCount > 1 ? "s" : ""} require immediate attention
        </span>
      </div>
      <button onClick={() => setVisible(false)} className="text-critical/60 hover:text-critical shrink-0 ml-2">
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
};

export default ThreatBanner;
