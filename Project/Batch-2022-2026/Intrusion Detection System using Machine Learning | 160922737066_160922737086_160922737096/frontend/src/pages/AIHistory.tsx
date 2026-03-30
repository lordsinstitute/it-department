import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Loader2, MessageSquare, Clock, Search, Trash2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/context/AuthContext";
import { useToast } from "@/hooks/use-toast";

interface AILog {
  id: string;
  question: string;
  answer: string;
  timestamp: string;
}

const AIHistory = () => {
  const [logs, setLogs] = useState<AILog[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const { user, isAdmin } = useAuth();
  const { toast } = useToast();

  const deleteLog = async (id: string) => {
    if (!confirm("Delete this AI log?")) return;
    const { error } = await supabase.from("ai_logs").delete().eq("id", id);
    if (error) { toast({ title: "Delete failed", description: error.message, variant: "destructive" }); return; }
    setLogs((prev) => prev.filter((l) => l.id !== id));
    toast({ title: "Log deleted" });
  };

  const deleteAllLogs = async () => {
    if (!user || !confirm("Delete ALL AI logs? This cannot be undone.")) return;
    const { error } = await supabase.from("ai_logs").delete().eq("user_id", user.id);
    if (error) { toast({ title: "Delete failed", description: error.message, variant: "destructive" }); return; }
    setLogs([]);
    toast({ title: "All AI logs deleted" });
  };

  useEffect(() => {
    const fetchLogs = async () => {
      if (!user) { setLoading(false); return; }
      const { data } = await supabase
        .from("ai_logs")
        .select("*")
        .eq("user_id", user.id)
        .order("timestamp", { ascending: false });
      setLogs(data || []);
      setLoading(false);
    };
    fetchLogs();
  }, [user]);

  const filtered = logs.filter((l) => {
    const q = search.toLowerCase();
    return !q || l.question.toLowerCase().includes(q) || l.answer.toLowerCase().includes(q);
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-3 text-sm text-muted-foreground">Loading AI history...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div>
            <h2 className="text-lg font-bold text-foreground">AI Chat History</h2>
            <p className="text-xs text-muted-foreground">{logs.length} conversations saved</p>
          </div>
          {isAdmin && logs.length > 0 && (
            <Button variant="destructive" size="sm" className="text-xs" onClick={deleteAllLogs}>
              <Trash2 className="mr-1.5 h-3.5 w-3.5" /> Delete All
            </Button>
          )}
        </div>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search history..." className="pl-9 bg-secondary/50 border-border text-xs" />
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="glass-card rounded-xl p-10 text-center">
          <MessageSquare className="mx-auto h-12 w-12 text-muted-foreground/20 mb-3" />
          <p className="text-sm text-muted-foreground">
            {logs.length === 0 ? "No AI conversations yet. Go to AI Assistant to start chatting." : "No results match your search."}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((log, i) => (
            <motion.div
              key={log.id}
              className="glass-card rounded-xl p-4"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-1 sm:gap-2 mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  <MessageSquare className="h-4 w-4 text-primary shrink-0" />
                  <p className="text-sm font-semibold text-foreground truncate">{log.question}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0 pl-6 sm:pl-0">
                  {isAdmin && (
                    <button
                      onClick={() => deleteLog(log.id)}
                      className="rounded p-1 text-muted-foreground hover:bg-destructive/20 hover:text-destructive transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                      {new Date(log.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-lg bg-secondary/30 p-3">
                <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">{log.answer}</p>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AIHistory;
