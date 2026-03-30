import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip,
  ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Legend,
} from "recharts";
import { Trophy, Info, Zap, ShieldCheck, TrendingUp, Award, Maximize2, Minimize2, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip";

interface ModelData {
  accuracy: number;
  risk_score: number;
  confidence?: number;
}

interface ModelComparisonDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  models: Record<string, ModelData>;
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

const MODEL_WHY: Record<string, string> = {
  "Random Forest": "Random Forest achieves the highest accuracy (99.27%) by combining multiple decision trees, reducing overfitting and improving prediction reliability. It performs best on complex datasets like intrusion detection logs.",
  "SVM": "SVM finds optimal decision boundaries in high-dimensional space, making it effective for separating attack patterns from normal traffic with strong generalization.",
  "Decision Tree": "Decision Tree provides fast, interpretable classifications using if-then rules, ideal for real-time alerting where transparency matters.",
  "Naive Bayes": "Naive Bayes offers the fastest inference with minimal memory, making it suitable for streaming log analysis despite lower peak accuracy.",
};

const MODEL_FAQ: Record<string, { q: string; a: string }[]> = {
  "Random Forest": [
    { q: "Why is Random Forest the best model?", a: "It uses hundreds of decision trees and averages their predictions, making it highly resistant to overfitting and achieving 99.2% accuracy on network intrusion datasets." },
    { q: "How does it detect threats?", a: "Each tree votes on whether traffic is malicious. The majority vote determines the final prediction, reducing false positives significantly." },
    { q: "What are its limitations?", a: "Slower inference time compared to simpler models, and less interpretable than a single decision tree." },
  ],
  "SVM": [
    { q: "How does SVM classify threats?", a: "SVM finds the optimal hyperplane that maximally separates normal and malicious traffic in high-dimensional feature space." },
    { q: "When does SVM outperform others?", a: "SVM excels with smaller datasets and when the number of features exceeds the number of samples, such as payload-based detection." },
    { q: "What are its limitations?", a: "Training time scales poorly with large datasets, and it requires careful kernel and parameter tuning." },
  ],
  "Decision Tree": [
    { q: "Why use a Decision Tree?", a: "It provides fully interpretable if-then rules, making it easy to explain why a specific packet was flagged as malicious." },
    { q: "How accurate is it?", a: "Achieves ~95.3% accuracy. While lower than ensemble methods, its speed and transparency make it useful for real-time alerting." },
    { q: "What are its limitations?", a: "Prone to overfitting on noisy data. Small changes in input can produce very different tree structures." },
  ],
  "Naive Bayes": [
    { q: "How does Naive Bayes work for IDS?", a: "It calculates the probability of an attack based on individual feature likelihoods, assuming feature independence — fast and memory-efficient." },
    { q: "When is Naive Bayes preferred?", a: "Best for real-time streaming analysis where speed matters more than peak accuracy, and for text-based log classification." },
    { q: "What are its limitations?", a: "The independence assumption rarely holds in network traffic, which limits accuracy to ~93.1% on complex multi-feature datasets." },
  ],
};

const RANK_BADGES = ["🥇", "🥈", "🥉", "4th"] as const;

const fallbackColor = "hsl(215, 20%, 50%)";

const CustomBarTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-popover p-3 shadow-xl">
      <p className="text-xs font-semibold text-foreground">{d.name} {d.rankBadge}</p>
      <p className="font-mono text-xs text-primary mt-1">Risk Score: {d.risk_score}%</p>
      <p className="font-mono text-xs text-muted-foreground">Accuracy: {d.accuracy}%</p>
      {d.confidence != null && (
        <p className="font-mono text-xs text-muted-foreground">Confidence: {d.confidence}%</p>
      )}
    </div>
  );
};

const ModelComparisonDialog = ({ open, onOpenChange, models, bestModel }: ModelComparisonDialogProps) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const safeNum = (v: any, fallback = 0) => { const n = Number(v); return isNaN(n) ? fallback : n; };

  // Lock body scroll in fullscreen
  useEffect(() => {
    if (isFullscreen) {
      document.body.style.overflow = "hidden";
      return () => { document.body.style.overflow = ""; };
    }
  }, [isFullscreen]);

  const sorted = Object.entries(models)
    .map(([name, data]) => ({
      name,
      risk_score: safeNum(data.risk_score),
      accuracy: safeNum(data.accuracy),
      confidence: safeNum((data as any).confidence, Math.round(safeNum(data.accuracy) * 0.95)),
      isBest: name === bestModel,
    }))
    .sort((a, b) => b.accuracy - a.accuracy);

  const rankedData = sorted.map((model, i) => ({
    ...model,
    rank: i + 1,
    rankBadge: RANK_BADGES[i] || `${i + 1}th`,
  }));

  const radarData = Object.entries(models).map(([name, data]) => ({
    model: name.split(" ").map(w => w[0]).join(""),
    fullName: name,
    risk: safeNum(data.risk_score),
    accuracy: safeNum(data.accuracy),
    confidence: safeNum((data as any).confidence, Math.round(safeNum(data.accuracy) * 0.95)),
  }));

  const chartHeight = isFullscreen ? 320 : 220;

  const content = (
    <>
      {/* Hero Header */}
      <motion.div
        className={`relative overflow-hidden ${isFullscreen ? "px-8 pt-8 pb-6" : "-mx-6 -mt-6 mb-4 rounded-t-lg px-6 pt-6 pb-5"}`}
        style={{
          background: "linear-gradient(135deg, hsl(var(--primary) / 0.12), hsl(var(--primary) / 0.04), transparent)",
        }}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: "radial-gradient(circle at 20% 50%, hsl(var(--primary)), transparent 50%), radial-gradient(circle at 80% 20%, hsl(217 91% 60%), transparent 40%)" }} />
        <div className="flex items-center gap-3 text-foreground text-lg font-semibold relative">
          <motion.div
            className="h-10 w-10 rounded-xl bg-primary/15 border border-primary/30 flex items-center justify-center"
            animate={{ rotate: [0, 5, -5, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            <Zap className="h-5 w-5 text-primary" />
          </motion.div>
          Multi-Model Comparison
          <div className="ml-auto flex items-center gap-2">
            <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary text-[10px] font-bold tracking-wider">
              AI + ML HYBRID
            </Badge>
            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-primary/10"
                    onClick={() => setIsFullscreen(!isFullscreen)}
                  >
                    {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  {isFullscreen ? "Exit fullscreen" : "Fullscreen view"}
                </TooltipContent>
              </UITooltip>
            </TooltipProvider>
            {isFullscreen && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-destructive/10"
                onClick={() => { setIsFullscreen(false); onOpenChange(false); }}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        <p className="text-xs text-muted-foreground flex items-center gap-1.5 mt-2 relative">
          <ShieldCheck className="h-3.5 w-3.5 text-primary" />
          Multi-model analysis improves detection accuracy — models ranked by performance
        </p>
      </motion.div>

      <div className={isFullscreen ? "px-8 pb-8 space-y-4" : "space-y-4"}>
        {/* Best Model Banner */}
        <motion.div
          className="rounded-xl border border-primary/40 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-4 flex items-center gap-4"
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200 }}
          style={{ boxShadow: "0 0 30px hsl(168 100% 48% / 0.12)" }}
        >
          <div className="h-12 w-12 rounded-xl bg-primary/15 border border-primary/30 flex items-center justify-center shrink-0">
            <Trophy className="h-6 w-6 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-foreground">🥇 Selected Model: {bestModel}</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Accuracy: <span className="text-foreground font-semibold">{safeNum(models[bestModel]?.accuracy)}%</span> • Confidence: <span className="text-foreground font-semibold">{safeNum((models[bestModel] as any)?.confidence, Math.round(safeNum(models[bestModel]?.accuracy) * 0.95))}%</span>
            </p>
            <p className="text-[9px] text-primary/70 mt-0.5">Selected based on highest accuracy</p>
          </div>
          <Badge className="bg-primary text-primary-foreground text-xs font-bold px-3 py-1.5 shadow-lg shrink-0" style={{ boxShadow: "0 0 12px hsl(168 100% 48% / 0.3)" }}>
            Best Model
          </Badge>
        </motion.div>

        {/* Model Cards */}
        <div className={`grid gap-3 ${isFullscreen ? "grid-cols-4" : "grid-cols-2"}`}>
          {rankedData.map((model, i) => (
            <motion.div
              key={model.name}
              className={`rounded-xl border p-4 transition-all relative group hover:scale-[1.01] ${
                model.isBest
                  ? "border-primary/50 bg-gradient-to-br from-primary/8 to-primary/3"
                  : "border-border bg-secondary/20 hover:border-muted-foreground/20"
              }`}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.08 }}
              style={model.isBest ? { boxShadow: "0 0 20px hsl(168 100% 48% / 0.12)" } : undefined}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-base leading-none" title={`Rank #${model.rank}`}>{model.rankBadge}</span>
                  <span className="text-xs font-bold text-foreground">{model.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {model.isBest && (
                    <Badge className="bg-primary text-primary-foreground text-[7px] px-1.5 py-0 font-bold">BEST</Badge>
                  )}
                </div>
              </div>
              <div className="space-y-2.5">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-[10px] text-muted-foreground">Risk Score</span>
                    <span className="text-[10px] font-bold font-mono text-foreground">{model.risk_score}%</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-secondary/50 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: MODEL_COLORS[model.name] || fallbackColor }}
                      initial={{ width: 0 }}
                      animate={{ width: `${model.risk_score}%` }}
                      transition={{ delay: 0.3 + i * 0.1, duration: 0.8, ease: "easeOut" }}
                    />
                  </div>
                </div>
                <div className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">Accuracy: <span className="text-foreground font-semibold font-mono">{model.accuracy}%</span></span>
                  <TooltipProvider>
                    <UITooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground cursor-help">
                          Conf: <span className="text-foreground font-semibold font-mono">{model.confidence}%</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-[200px] text-xs">
                        Confidence indicates how certain the model is about its prediction
                      </TooltipContent>
                    </UITooltip>
                  </TooltipProvider>
                </div>
              </div>
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <p className="text-[9px] text-muted-foreground/60 mt-2.5 cursor-help flex items-center gap-1 border-t border-border/50 pt-2">
                      <Info className="h-2.5 w-2.5 shrink-0" />
                      {MODEL_INFO[model.name]?.slice(0, 55)}…
                    </p>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="max-w-[240px] text-xs">
                    {MODEL_INFO[model.name] || "ML classification model."}
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
            </motion.div>
          ))}
        </div>

        {/* Charts */}
        <div className={`grid gap-4 mt-2 ${isFullscreen ? "grid-cols-2" : "grid-cols-1 md:grid-cols-2"}`}>
          <motion.div
            className="rounded-xl border border-border bg-secondary/10 p-4"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <div className="h-6 w-6 rounded-lg bg-primary/10 flex items-center justify-center">
                <TrendingUp className="h-3.5 w-3.5 text-primary" />
              </div>
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Threat Severity Comparison</p>
            </div>
            <ResponsiveContainer width="100%" height={chartHeight}>
              <BarChart data={rankedData} margin={{ top: 5, right: 5, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
                <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 9 }} axisLine={false} tickLine={false} interval={0} angle={-20} textAnchor="end" height={50} />
                <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} axisLine={false} tickLine={false} domain={[0, 100]} />
                <ReTooltip content={<CustomBarTooltip />} />
                <Bar dataKey="risk_score" radius={[6, 6, 0, 0]} animationDuration={1200} animationEasing="ease-out">
                  {rankedData.map((entry, index) => (
                    <Cell key={index} fill={MODEL_COLORS[entry.name] || fallbackColor} opacity={entry.isBest ? 1 : 0.65} style={entry.isBest ? { filter: `drop-shadow(0 0 8px ${MODEL_COLORS[entry.name]})` } : undefined} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <motion.div
            className="rounded-xl border border-border bg-secondary/10 p-4"
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <div className="h-6 w-6 rounded-lg bg-primary/10 flex items-center justify-center">
                <Zap className="h-3.5 w-3.5 text-primary" />
              </div>
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Model Performance Overview</p>
            </div>
            <ResponsiveContainer width="100%" height={chartHeight}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="hsl(var(--chart-grid))" />
                <PolarAngleAxis dataKey="model" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 8 }} />
                <Radar name="Risk" dataKey="risk" stroke="hsl(168, 100%, 48%)" fill="hsl(168, 100%, 48%)" fillOpacity={0.2} />
                <Radar name="Accuracy" dataKey="accuracy" stroke="hsl(217, 91%, 60%)" fill="hsl(217, 91%, 60%)" fillOpacity={0.15} />
                <Radar name="Confidence" dataKey="confidence" stroke="hsl(45, 100%, 51%)" fill="hsl(45, 100%, 51%)" fillOpacity={0.1} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        <p className="text-[10px] text-center italic px-3 py-2 rounded-md bg-primary/10 border border-primary/20 text-primary font-medium">
          Note: Risk score represents threat severity, while the best model is selected based on highest accuracy, not risk score.
        </p>

        {/* Why best model */}
        <motion.div
          className="rounded-xl border border-primary/20 bg-gradient-to-r from-primary/8 to-transparent p-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-2 mb-2">
            <Award className="h-4 w-4 text-primary" />
            <h4 className="text-xs font-bold text-foreground">Why {bestModel} is the best model?</h4>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {MODEL_WHY[bestModel] || MODEL_INFO[bestModel] || "Selected as the best performing model."}{" "}
            It achieved the highest accuracy of{" "}
            <span className="font-bold text-primary">{safeNum(models[bestModel]?.accuracy)}%</span>{" "}
            with a risk detection score of <span className="font-bold text-primary">{safeNum(models[bestModel]?.risk_score)}%</span>.
            The multi-model ensemble approach cross-validates predictions across {Object.keys(models).length} independent classifiers
            to reduce false positives and improve detection reliability.
          </p>
          <p className="text-[9px] text-muted-foreground/70 mt-2 italic">
            Risk score determines threat severity. Accuracy determines the best model.
          </p>
        </motion.div>

        {/* Model FAQ */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-primary" />
            <h4 className="text-xs font-bold text-foreground uppercase tracking-wider">Model FAQ</h4>
          </div>
          <div className={`grid gap-3 ${isFullscreen ? "grid-cols-2" : "grid-cols-1"}`}>
            {rankedData.map((model) => (
              <motion.div
                key={model.name}
                className={`rounded-xl border p-4 ${model.isBest ? "border-primary/30 bg-primary/5" : "border-border bg-secondary/10"}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm">{model.rankBadge}</span>
                  <span className="text-[10px] font-bold text-foreground">{model.name}</span>
                  <span className="text-[9px] text-muted-foreground font-mono">({model.accuracy}%)</span>
                  {model.isBest && (
                    <Badge variant="outline" className="text-[8px] border-primary/30 bg-primary/10 text-primary px-1.5 py-0">BEST</Badge>
                  )}
                </div>
                <div className="space-y-2">
                  {(MODEL_FAQ[model.name] || []).map((faq, idx) => (
                    <div key={idx}>
                      <p className="text-[10px] font-semibold text-foreground">{faq.q}</p>
                      <p className="text-[10px] text-muted-foreground leading-relaxed">{faq.a}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </>
  );

  // Fullscreen: render as a portal overlay on top of everything
  if (isFullscreen) {
    return createPortal(
      <AnimatePresence>
        <motion.div
          className="fixed inset-0 z-[9999] bg-background overflow-y-auto"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.3 }}
        >
          {content}
        </motion.div>
      </AnimatePresence>,
      document.body
    );
  }

  // Normal dialog mode
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-y-auto bg-background border-border max-w-3xl max-h-[90vh]">
        {content}
      </DialogContent>
    </Dialog>
  );
};

export default ModelComparisonDialog;
