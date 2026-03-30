import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Loader2, Shield, Users, AlertTriangle, MessageSquare, Clock, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { supabase } from "@/lib/supabase";

interface AlertRow {
  id: string;
  prediction: string;
  severity: string;
  risk_score: number;
  timestamp: string;
  user_id: string;
  source_ip: string;
  status: string;
}

interface AILogRow {
  id: string;
  question: string;
  answer: string;
  timestamp: string;
  user_id: string;
}

interface UserRow {
  id: string;
  email: string;
  created_at: string;
}

interface UserRoleRow {
  user_id: string;
  role: string;
}

const AdminDashboard = () => {
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [aiLogs, setAILogs] = useState<AILogRow[]>([]);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [userRoles, setUserRoles] = useState<UserRoleRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [roleUpdating, setRoleUpdating] = useState<string | null>(null);

  useEffect(() => {
    const fetchAll = async () => {
      const [alertsRes, logsRes, profilesRes, rolesRes] = await Promise.all([
        supabase.from("alerts").select("*").order("timestamp", { ascending: false }),
        supabase.from("ai_logs").select("*").order("timestamp", { ascending: false }),
        supabase.from("profiles").select("id, email, created_at").order("created_at", { ascending: false }),
        supabase.from("user_roles").select("user_id, role"),
      ]);
      setAlerts(alertsRes.data || []);
      setAILogs(logsRes.data || []);
      setUsers(profilesRes.data || []);
      setUserRoles(rolesRes.data || []);
      setLoading(false);
    };
    fetchAll();
  }, []);

  const getUserRole = (userId: string) => {
    return userRoles.find((r) => r.user_id === userId)?.role || "user";
  };

  const toggleRole = async (userId: string) => {
    setRoleUpdating(userId);
    const currentRole = getUserRole(userId);
    const newRole = currentRole === "admin" ? "user" : "admin";

    const existing = userRoles.find((r) => r.user_id === userId);
    if (existing) {
      await supabase.from("user_roles").update({ role: newRole }).eq("user_id", userId);
    } else {
      await supabase.from("user_roles").insert({ user_id: userId, role: newRole });
    }

    setUserRoles((prev) => {
      const filtered = prev.filter((r) => r.user_id !== userId);
      return [...filtered, { user_id: userId, role: newRole }];
    });
    setRoleUpdating(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-3 text-sm text-muted-foreground">Loading admin data...</span>
      </div>
    );
  }

  const filteredAlerts = alerts.filter((a) => {
    const q = search.toLowerCase();
    return !q || a.prediction?.toLowerCase().includes(q) || a.source_ip?.includes(q) || a.severity?.toLowerCase().includes(q);
  });

  const filteredLogs = aiLogs.filter((l) => {
    const q = search.toLowerCase();
    return !q || l.question?.toLowerCase().includes(q) || l.answer?.toLowerCase().includes(q);
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-critical/10 p-2">
            <Shield className="h-5 w-5 text-critical" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-foreground">Admin Dashboard</h2>
            <p className="text-xs text-muted-foreground">Full system overview — all users</p>
          </div>
        </div>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search..." className="pl-9 bg-secondary/50 border-border text-xs" />
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Total Users", value: users.length, icon: Users, color: "text-primary" },
          { label: "Total Alerts", value: alerts.length, icon: AlertTriangle, color: "text-critical" },
          { label: "AI Conversations", value: aiLogs.length, icon: MessageSquare, color: "text-warning" },
          { label: "Admins", value: userRoles.filter((r) => r.role === "admin").length, icon: Shield, color: "text-success" },
        ].map((s) => (
          <div key={s.label} className="glass-card rounded-xl px-4 py-3 text-center">
            <s.icon className={`h-5 w-5 mx-auto mb-1 ${s.color}`} />
            <p className={`font-mono text-xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </div>

      <Tabs defaultValue="users" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-secondary/50">
          <TabsTrigger value="users" className="text-xs">Users ({users.length})</TabsTrigger>
          <TabsTrigger value="alerts" className="text-xs">All Alerts ({alerts.length})</TabsTrigger>
          <TabsTrigger value="ai" className="text-xs">AI Logs ({aiLogs.length})</TabsTrigger>
        </TabsList>

        {/* Users Tab */}
        <TabsContent value="users" className="space-y-2 mt-3">
          {users.map((u, i) => (
            <motion.div
              key={u.id}
              className="glass-card rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-secondary shrink-0">
                  <Users className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{u.email}</p>
                  <div className="flex items-center gap-2">
                    <Clock className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground">
                      Joined {new Date(u.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`inline-flex rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                  getUserRole(u.id) === "admin" ? "bg-critical/15 text-critical" : "bg-primary/15 text-primary"
                }`}>
                  {getUserRole(u.id).toUpperCase()}
                </span>
                <button
                  onClick={() => toggleRole(u.id)}
                  disabled={roleUpdating === u.id}
                  className="rounded-lg border border-border bg-secondary/50 px-3 py-1.5 text-[10px] font-medium text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors disabled:opacity-50"
                >
                  {roleUpdating === u.id ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : getUserRole(u.id) === "admin" ? (
                    "Demote to User"
                  ) : (
                    "Promote to Admin"
                  )}
                </button>
              </div>
            </motion.div>
          ))}
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-2 mt-3">
          {filteredAlerts.length === 0 ? (
            <div className="glass-card rounded-xl p-10 text-center">
              <p className="text-sm text-muted-foreground">No alerts found.</p>
            </div>
          ) : (
            filteredAlerts.slice(0, 50).map((a, i) => (
              <motion.div
                key={a.id}
                className="glass-card rounded-xl p-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.02 }}
              >
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <AlertTriangle className={`h-4 w-4 shrink-0 ${
                      a.severity === "Critical" ? "text-critical" : a.severity === "High" ? "text-high" : "text-warning"
                    }`} />
                    <span className="text-sm font-medium text-foreground truncate">{a.prediction}</span>
                    <span className={`inline-flex rounded-full px-2 py-0.5 text-[9px] font-bold ${
                      a.severity === "Critical" ? "bg-critical/15 text-critical" :
                      a.severity === "High" ? "bg-high/15 text-high" :
                      "bg-warning/15 text-warning"
                    }`}>{a.severity}</span>
                  </div>
                  <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                    <span>Risk: {a.risk_score}</span>
                    <span>{a.source_ip}</span>
                    <span>{new Date(a.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </TabsContent>

        {/* AI Logs Tab */}
        <TabsContent value="ai" className="space-y-2 mt-3">
          {filteredLogs.length === 0 ? (
            <div className="glass-card rounded-xl p-10 text-center">
              <p className="text-sm text-muted-foreground">No AI logs found.</p>
            </div>
          ) : (
            filteredLogs.slice(0, 50).map((l, i) => (
              <motion.div
                key={l.id}
                className="glass-card rounded-xl p-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.02 }}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-primary shrink-0" />
                    <p className="text-sm font-semibold text-foreground">{l.question}</p>
                  </div>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {new Date(l.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="rounded-lg bg-secondary/30 p-3">
                  <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap line-clamp-3">{l.answer}</p>
                </div>
              </motion.div>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdminDashboard;
