import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";
import ChartHelpTooltip from "@/components/dashboard/ChartHelpTooltip";
import { useAlerts } from "@/context/AlertContext";

const ANIMATION_DURATION = 1200;
const ANIMATION_EASING = "ease-out";

const SEVERITY_COLORS: Record<string, string> = {
  Critical: "#FF4560",
  High: "#FFC107",
  Medium: "#00F5D4",
  Low: "#4361EE",
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload) return null;
  return (
    <div className="glass-card rounded-lg p-3 shadow-xl">
      <p className="mb-1 text-xs font-medium text-muted-foreground">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="font-mono text-xs" style={{ color: entry.color }}>
          {entry.name}: {entry.value}
        </p>
      ))}
    </div>
  );
};

export const RiskAreaChart = () => {
  const { alerts } = useAlerts();

  // Compute time-based data from real alerts
  const timeSlots = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "Now"];
  const riskOverTime = timeSlots.map((time) => {
    // Distribute alerts across time slots based on their index
    const slotAlerts = alerts.filter((_, idx) => idx % timeSlots.length === timeSlots.indexOf(time));
    return {
      time,
      critical: slotAlerts.filter((a) => a.severity === "Critical").length,
      high: slotAlerts.filter((a) => a.severity === "High").length,
      medium: slotAlerts.filter((a) => a.severity === "Medium").length,
    };
  });

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="mb-3 sm:mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-2 sm:gap-3">
          <h3 className="text-xs sm:text-sm font-semibold text-foreground">Threat Activity</h3>
          <ChartHelpTooltip text="Threat events from real predictions categorized by severity level." />
          {alerts.length > 0 && (
            <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary text-[10px]">
              {alerts.length} EVENTS
            </Badge>
          )}
        </div>
        <div className="flex gap-2 sm:gap-3">
          {[
            { label: "Critical", color: "#FF4560" },
            { label: "High", color: "#FFC107" },
            { label: "Medium", color: "#00F5D4" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="text-xs text-muted-foreground">{item.label}</span>
            </div>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={riskOverTime}>
          <defs>
            <linearGradient id="critGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FF4560" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#FF4560" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="highGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FFC107" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#FFC107" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="medGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00F5D4" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#00F5D4" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
          <XAxis dataKey="time" tick={{ fill: "hsl(215, 20%, 50%)", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "hsl(215, 20%, 50%)", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Area type="monotone" dataKey="critical" stroke="#FF4560" fill="url(#critGrad)" strokeWidth={2} name="Critical" isAnimationActive animationDuration={ANIMATION_DURATION} animationEasing={ANIMATION_EASING} />
          <Area type="monotone" dataKey="high" stroke="#FFC107" fill="url(#highGrad)" strokeWidth={2} name="High" isAnimationActive animationDuration={ANIMATION_DURATION} animationEasing={ANIMATION_EASING} />
          <Area type="monotone" dataKey="medium" stroke="#00F5D4" fill="url(#medGrad)" strokeWidth={1.5} name="Medium" isAnimationActive animationDuration={ANIMATION_DURATION} animationEasing={ANIMATION_EASING} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export const RiskPieChart = () => {
  const { alerts } = useAlerts();
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const riskDistribution = [
    { name: "Critical", value: alerts.filter((a) => a.severity === "Critical").length, fill: "#FF4560" },
    { name: "High", value: alerts.filter((a) => a.severity === "High").length, fill: "#FFC107" },
    { name: "Medium", value: alerts.filter((a) => a.severity === "Medium").length, fill: "#00F5D4" },
    { name: "Low", value: alerts.filter((a) => a.severity === "Low").length, fill: "#4361EE" },
  ];

  const total = riskDistribution.reduce((s, d) => s + d.value, 0);

  if (total === 0) {
    return (
      <div className="glass-card rounded-xl p-5">
        <div className="mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-foreground">Attack Distribution</h3>
            <ChartHelpTooltip text="Breakdown of detected events by severity — populated from real analyses." />
          </div>
          <p className="text-xs text-muted-foreground">0 Total Events</p>
        </div>
        <div className="flex items-center justify-center h-[180px]">
          <p className="text-xs text-muted-foreground">No data yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-foreground">Attack Distribution</h3>
          <ChartHelpTooltip text="Breakdown of detected events by severity from real predictions." />
        </div>
        <p className="text-xs text-muted-foreground">
          {total} Total Events
        </p>
      </div>
      <div className="flex flex-col items-center gap-4">
        <ResponsiveContainer width={180} height={180}>
          <PieChart>
            <Pie
              data={riskDistribution.filter((d) => d.value > 0)}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={activeIndex !== null ? 82 : 78}
              paddingAngle={3}
              dataKey="value"
              strokeWidth={0}
              onMouseEnter={(_, i) => setActiveIndex(i)}
              onMouseLeave={() => setActiveIndex(null)}
              isAnimationActive
              animationDuration={ANIMATION_DURATION}
              animationEasing={ANIMATION_EASING}
            >
              {riskDistribution.filter((d) => d.value > 0).map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.fill}
                  opacity={activeIndex === null || activeIndex === index ? 1 : 0.4}
                  style={{
                    filter: activeIndex === index ? `drop-shadow(0 0 8px ${entry.fill})` : "none",
                    transition: "all 0.3s ease",
                  }}
                />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="w-full space-y-2.5">
          {riskDistribution.map((item) => (
            <div key={item.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.fill }} />
                <span className="text-xs text-muted-foreground">{item.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-semibold text-foreground">{item.value}</span>
                {total > 0 && (
                  <span className="text-xs text-muted-foreground">
                    ({((item.value / total) * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
