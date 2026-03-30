import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Trophy, Info, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { ModelScore } from "@/lib/api";

interface ModelComparisonProps {
  models: Record<string, ModelScore>;
  bestModel: string;
}

const MODEL_COLORS: Record<string, string> = {
  "Random Forest": "hsl(168, 100%, 48%)",
  "SVM": "hsl(217, 91%, 60%)",
  "Decision Tree": "hsl(45, 100%, 51%)",
  "Naive Bayes": "hsl(270, 80%, 60%)",
};

const MODEL_INFO: Record<string, string> = {
  "Random Forest": "Ensemble learning with multiple decision trees — best accuracy on complex datasets.",
  "SVM": "Support Vector Machine — effective for high-dimensional feature spaces.",
  "Decision Tree": "Single tree classifier — fast and interpretable but may overfit.",
  "Naive Bayes": "Probabilistic classifier — efficient with strong independence assumptions.",
};

const fallbackColor = "hsl(215, 20%, 50%)";

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="glass-card rounded-lg p-3 shadow-xl border border-border">
      <p className="text-xs font-semibold text-foreground">{d.name}</p>
      <p className="font-mono text-xs text-primary mt-1">Risk Score: {d.risk_score}</p>
      <p className="font-mono text-xs text-muted-foreground">Accuracy: {d.accuracy}%</p>
    </div>
  );
};

const ModelComparison = ({ models, bestModel }: ModelComparisonProps) => {
  const chartData = Object.entries(models).map(([name, data]) => ({
    name,
    risk_score: data.risk_score,
    accuracy: data.accuracy,
    isBest: name === bestModel,
  }));

  return (
    <motion.div
      className="glass-card rounded-xl p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Model Comparison</h3>
          <TooltipProvider>
            <UITooltip>
              <TooltipTrigger asChild>
                <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[260px] text-xs">
                Random Forest performs best due to ensemble learning and better accuracy in complex datasets.
              </TooltipContent>
            </UITooltip>
          </TooltipProvider>
        </div>
        <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary text-[10px]">
          {Object.keys(models).length} MODELS
        </Badge>
      </div>

      <p className="text-[10px] text-muted-foreground mb-4 flex items-center gap-1">
        <Zap className="h-3 w-3" />
        Multi-model analysis improves detection accuracy
      </p>

      {/* Selected Model Badge */}
      <motion.div
        className="mb-4 rounded-lg border border-primary/40 bg-primary/10 p-3 flex items-center gap-3"
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
        style={{ boxShadow: "0 0 20px hsl(168 100% 48% / 0.15)" }}
      >
        <Trophy className="h-5 w-5 text-primary shrink-0" />
        <div className="flex-1">
          <p className="text-xs font-bold text-foreground">Selected Model: {bestModel}</p>
          <p className="text-[10px] text-muted-foreground">
            Accuracy: {models[bestModel]?.accuracy ?? "N/A"}%
          </p>
        </div>
        <Badge className="bg-primary text-primary-foreground text-[9px] font-bold">
          Best Model
        </Badge>
      </motion.div>

      {/* Model Cards */}
      <div className="grid grid-cols-2 gap-2 mb-5">
        {chartData.map((model, i) => (
          <motion.div
            key={model.name}
            className={`rounded-lg border p-3 transition-all ${
              model.isBest
                ? "border-primary/50 bg-primary/5"
                : "border-border bg-secondary/20"
            }`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + i * 0.08 }}
            style={model.isBest ? { boxShadow: "0 0 12px hsl(168 100% 48% / 0.12)" } : undefined}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-semibold text-foreground truncate">{model.name}</span>
              {model.isBest && (
                <Trophy className="h-3 w-3 text-primary shrink-0" />
              )}
            </div>
            {/* Risk score bar */}
            <div className="h-1.5 rounded-full bg-secondary/50 overflow-hidden mb-1">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: MODEL_COLORS[model.name] || fallbackColor }}
                initial={{ width: 0 }}
                animate={{ width: `${model.risk_score}%` }}
                transition={{ delay: 0.4 + i * 0.1, duration: 0.8, ease: "easeOut" }}
              />
            </div>
            <div className="flex justify-between">
              <span className="text-[9px] text-muted-foreground">Risk: {model.risk_score}%</span>
              <span className="text-[9px] text-muted-foreground">Acc: {model.accuracy}%</span>
            </div>
            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger asChild>
                  <p className="text-[8px] text-muted-foreground/60 mt-1 cursor-help truncate">
                    {MODEL_INFO[model.name]?.slice(0, 50)}…
                  </p>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[220px] text-xs">
                  {MODEL_INFO[model.name] || "ML classification model."}
                </TooltipContent>
              </UITooltip>
            </TooltipProvider>
          </motion.div>
        ))}
      </div>

      {/* Bar Chart */}
      <div className="rounded-lg border border-border bg-secondary/10 p-3">
        <p className="text-[10px] font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Risk Score Comparison</p>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
            <XAxis
              dataKey="name"
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 9 }}
              axisLine={false}
              tickLine={false}
              interval={0}
              angle={-15}
              textAnchor="end"
              height={40}
            />
            <YAxis
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="risk_score"
              radius={[4, 4, 0, 0]}
              isAnimationActive
              animationDuration={1200}
              animationEasing="ease-out"
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={MODEL_COLORS[entry.name] || fallbackColor}
                  opacity={entry.isBest ? 1 : 0.7}
                  style={entry.isBest ? { filter: `drop-shadow(0 0 6px ${MODEL_COLORS[entry.name]})` } : undefined}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};

export default ModelComparison;
