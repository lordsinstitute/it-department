import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from "react";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/context/AuthContext";

export interface StoredAlert {
  id: string;
  timestamp: string;
  prediction: string;
  risk_score: number;
  explanation: string;
  severity: "Critical" | "High" | "Medium" | "Low";
  sourceIP: string;
  status: "Active" | "Investigating" | "Resolved";
  inputSnippet: string;
}

const mapSeverity = (s: string): StoredAlert["severity"] => {
  const l = s?.toLowerCase() || "";
  if (l === "critical") return "Critical";
  if (l === "high") return "High";
  if (l === "medium") return "Medium";
  return "Low";
};

const mapStatus = (s: string): StoredAlert["status"] => {
  const l = s?.toLowerCase() || "";
  if (l === "active") return "Active";
  if (l === "resolved") return "Resolved";
  return "Investigating";
};

const dbToAlert = (row: any): StoredAlert => ({
  id: row.id,
  timestamp: (row.timestamp || row.created_at || "").replace("T", " ").slice(0, 19),
  prediction: row.prediction || "",
  risk_score: row.risk_score ?? 0,
  explanation: row.explanation || "",
  severity: mapSeverity(row.severity),
  sourceIP: row.source_ip || row.sourceIP || "",
  status: mapStatus(row.status),
  inputSnippet: row.input_snippet || "",
});

interface AlertContextValue {
  alerts: StoredAlert[];
  loading: boolean;
  error: string | null;
  refreshAlerts: () => Promise<void>;
  deleteAlert: (id: string) => Promise<void>;
  deleteAllAlerts: () => Promise<void>;
  totalAlerts: number;
  criticalCount: number;
  highCount: number;
  resolvedCount: number;
  activeCount: number;
  averageRiskScore: number;
}

const AlertContext = createContext<AlertContextValue | null>(null);

export const AlertProvider = ({ children }: { children: ReactNode }) => {
  const [alerts, setAlerts] = useState<StoredAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();

  const refreshAlerts = useCallback(async () => {
    if (!user) { setLoading(false); return; }
    try {
      const { data, error: err } = await supabase
        .from("alerts")
        .select("*")
        .eq("user_id", user.id)
        .order("timestamp", { ascending: false });

      if (err) throw err;
      setAlerts((data || []).map(dbToAlert));
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to fetch alerts");
    } finally {
      setLoading(false);
    }
  }, [user]);

  // Initial fetch
  useEffect(() => {
    refreshAlerts();
  }, [refreshAlerts]);

  // Real-time subscription
  useEffect(() => {
    const channel = supabase
      .channel("alerts-realtime")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "alerts" },
        () => {
          // Re-fetch all alerts on any change
          refreshAlerts();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [refreshAlerts]);

  const deleteAlert = useCallback(async (id: string) => {
    const { error } = await supabase.from("alerts").delete().eq("id", id);
    if (error) throw new Error(error.message);
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const deleteAllAlerts = useCallback(async () => {
    if (!user) return;
    const { error } = await supabase.from("alerts").delete().eq("user_id", user.id);
    if (error) throw new Error(error.message);
    setAlerts([]);
  }, [user]);

  const totalAlerts = alerts.length;
  const criticalCount = alerts.filter((a) => a.severity === "Critical").length;
  const highCount = alerts.filter((a) => a.severity === "High").length;
  const resolvedCount = alerts.filter((a) => a.status === "Resolved").length;
  const activeCount = alerts.filter((a) => a.status === "Active").length;
  const averageRiskScore = totalAlerts > 0
    ? Math.round(alerts.reduce((s, a) => s + a.risk_score, 0) / totalAlerts * 10) / 10
    : 0;

  return (
    <AlertContext.Provider value={{
      alerts, loading, error, refreshAlerts, deleteAlert, deleteAllAlerts,
      totalAlerts, criticalCount, highCount, resolvedCount, activeCount, averageRiskScore,
    }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlerts = () => {
  const ctx = useContext(AlertContext);
  if (!ctx) throw new Error("useAlerts must be used within AlertProvider");
  return ctx;
};
