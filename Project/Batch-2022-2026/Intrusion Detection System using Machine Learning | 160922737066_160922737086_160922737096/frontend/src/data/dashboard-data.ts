// ── Static offline training data (kept for Performance page) ──

export const modelPerformance = [
  { name: "Random Forest", accuracy: 99.27, color: "#00F5D4" },
  { name: "Gradient Boost", accuracy: 96.90, color: "#22C55E" },
  { name: "Decision Tree", accuracy: 94.20, color: "#4361EE" },
  { name: "SVM", accuracy: 93.50, color: "#A855F7" },
  { name: "KNN", accuracy: 91.80, color: "#6B7280" },
  { name: "Naive Bayes", accuracy: 88.40, color: "#FF8C00" },
];

export const shapFeatures = [
  { feature: "src_bytes", importance: 0.183, color: "#FF4560" },
  { feature: "count", importance: 0.132, color: "#FFC107" },
  { feature: "srv_count", importance: 0.118, color: "#00F5D4" },
  { feature: "duration", importance: 0.094, color: "#4361EE" },
  { feature: "dst_host_count", importance: 0.069, color: "#A855F7" },
  { feature: "logged_in", importance: 0.055, color: "#22C55E" },
];

export const confusionMatrix = {
  truePositive: 4821,
  falsePositive: 35,
  falseNegative: 12,
  trueNegative: 5132,
};
