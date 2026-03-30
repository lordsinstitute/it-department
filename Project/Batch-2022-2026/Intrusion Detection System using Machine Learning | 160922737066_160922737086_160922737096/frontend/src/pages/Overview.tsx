import { motion } from "framer-motion";
import KpiCard from "@/components/dashboard/KpiCard";
import { RiskAreaChart, RiskPieChart } from "@/components/dashboard/RiskCharts";
import AlertsTable from "@/components/dashboard/AlertsTable";
import AttackTypeBreakdown from "@/components/dashboard/AttackTypeBreakdown";
import QuickActions from "@/components/dashboard/QuickActions";
import ThreatIntelFeed from "@/components/dashboard/ThreatIntelFeed";
import DashboardAIPanel from "@/components/dashboard/DashboardAIPanel";
import { useAlerts } from "@/context/AlertContext";
import { Scan, AlertOctagon, ShieldCheck, Shield, Loader2, WifiOff } from "lucide-react";

const Overview = () => {
  const { totalAlerts, criticalCount, resolvedCount, averageRiskScore, loading, error } = useAlerts();

  const kpiData = [
    {
      title: "Total Scans",
      value: totalAlerts.toLocaleString(),
      change: totalAlerts > 0 ? `${totalAlerts} total` : "—",
      trend: "up" as const,
      icon: Scan,
      color: "blue" as const,
      sparkline: [0, 0, 0, 0, 0, 0, totalAlerts],
      subtitle: "from real analyses",
    },
    {
      title: "Critical Threats",
      value: String(criticalCount),
      change: criticalCount > 0 ? "require action" : "none",
      trend: criticalCount > 0 ? "up" as const : "down" as const,
      icon: AlertOctagon,
      color: "critical" as const,
      sparkline: [0, 0, 0, 0, 0, 0, criticalCount],
      subtitle: "require immediate action",
    },
    {
      title: "Resolved",
      value: String(resolvedCount),
      change: resolvedCount > 0 ? `${resolvedCount} resolved` : "—",
      trend: "up" as const,
      icon: ShieldCheck,
      color: "primary" as const,
      sparkline: [0, 0, 0, 0, 0, 0, resolvedCount],
      subtitle: "threats resolved",
    },
    {
      title: "Avg Risk Score",
      value: String(averageRiskScore),
      change: averageRiskScore > 0 ? `avg of ${totalAlerts}` : "—",
      trend: averageRiskScore > 50 ? "up" as const : "down" as const,
      icon: Shield,
      color: "purple" as const,
      sparkline: [0, 0, 0, 0, 0, 0, averageRiskScore],
      subtitle: "average risk score",
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-3 text-sm text-muted-foreground">Loading alerts from backend...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {error && (
        <motion.div className="glass-card rounded-xl p-4 border border-critical/30 text-center flex items-center justify-center gap-2" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <WifiOff className="h-4 w-4 text-critical" />
          <p className="text-xs text-critical font-medium">Backend not reachable: {error}</p>
        </motion.div>
      )}

      {!error && totalAlerts === 0 && (
        <motion.div className="glass-card rounded-xl p-4 border border-primary/30 text-center" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <p className="text-xs text-muted-foreground">
            No data yet — go to <span className="text-primary font-semibold">Analyze Threats</span> to run your first analysis.
          </p>
        </motion.div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        {kpiData.map((kpi, i) => (
          <motion.div key={kpi.title} initial={{ opacity: 0, y: 30, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ delay: i * 0.1, duration: 0.5 }}>
            <KpiCard {...kpi} />
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-3 sm:gap-4 lg:grid-cols-3">
        <motion.div className="lg:col-span-2" initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
          <RiskAreaChart />
        </motion.div>
        <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}>
          <RiskPieChart />
        </motion.div>
      </div>

      {/* AI Assistant Panel (replaces map) */}
      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
        <DashboardAIPanel />
      </motion.div>

      {/* Attack Breakdown + Quick Actions */}
      <div className="grid grid-cols-1 gap-3 sm:gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <AttackTypeBreakdown />
        </div>
        <QuickActions />
      </div>

      {/* Alerts Table */}
      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
        <AlertsTable />
      </motion.div>

      <ThreatIntelFeed />
    </div>
  );
};

export default Overview;
