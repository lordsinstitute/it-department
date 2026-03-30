import { TrendingUp, TrendingDown, LucideIcon } from "lucide-react";

interface KpiCardProps {
  title: string;
  value: string;
  change: string;
  trend: "up" | "down";
  icon: LucideIcon;
  color: "primary" | "critical" | "blue" | "purple";
  sparkline: number[];
  subtitle?: string;
}

const colorMap = {
  primary: {
    text: "text-primary",
    glow: "glow-primary",
    stroke: "#00F5D4",
    fill: "rgba(0,245,212,0.1)",
  },
  critical: {
    text: "text-critical",
    glow: "glow-critical",
    stroke: "#FF4560",
    fill: "rgba(255,69,96,0.1)",
  },
  blue: {
    text: "text-[hsl(217,91%,60%)]",
    glow: "glow-blue",
    stroke: "#4361EE",
    fill: "rgba(67,97,238,0.1)",
  },
  purple: {
    text: "text-[hsl(270,80%,60%)]",
    glow: "glow-purple",
    stroke: "#A855F7",
    fill: "rgba(168,85,247,0.1)",
  },
};

const MiniSparkline = ({ data, color, fill }: { data: number[]; color: string; fill: string }) => {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const w = 100;
  const h = 32;
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(" ");
  const areaPoints = `0,${h} ${points} ${w},${h}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-8 w-full" preserveAspectRatio="none">
      <polygon points={areaPoints} fill={fill} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
};

const KpiCard = ({ title, value, change, trend, icon: Icon, color, sparkline, subtitle }: KpiCardProps) => {
  const c = colorMap[color];

  return (
    <div className={`glass-card relative overflow-hidden rounded-xl p-3 sm:p-5 transition-all hover:scale-[1.02] ${c.glow}`}>
      <div className="flex items-start justify-between">
        <div className="space-y-0.5 sm:space-y-1">
          <p className="text-[10px] sm:text-xs font-medium uppercase tracking-wider text-muted-foreground">{title}</p>
          <p className={`text-xl sm:text-3xl font-bold tracking-tight ${c.text}`}>{value}</p>
          {subtitle && <p className="text-[9px] sm:text-[10px] text-muted-foreground">{subtitle}</p>}
        </div>
        <div className="rounded-lg bg-secondary/60 p-1.5 sm:p-2">
          <Icon className={`h-4 w-4 sm:h-5 sm:w-5 ${c.text}`} />
        </div>
      </div>
      <div className="mt-3">
        <MiniSparkline data={sparkline} color={c.stroke} fill={c.fill} />
      </div>
      <div className="mt-2 flex items-center gap-1.5">
        {trend === "up" ? (
          <TrendingUp className="h-3 w-3 text-success" />
        ) : (
          <TrendingDown className="h-3 w-3 text-critical" />
        )}
        <span className="font-mono text-xs text-muted-foreground">{change}</span>
        <span className="text-xs text-muted-foreground">vs last week</span>
      </div>
    </div>
  );
};

export default KpiCard;
