import { useState } from "react";
import { Bot, Send, Loader2, Mic, MicOff } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { askAI as askAIApi } from "@/lib/api";

const DashboardAIPanel = () => {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleAsk = async (text: string) => {
    if (!text.trim()) return;
    setLoading(true);
    setResponse("");
    try {
      const answer = await askAIApi(text.trim());
      setResponse(answer);
      toast({ title: "AI response received" });
    } catch (err: any) {
      setResponse(`⚠️ ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleMic = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    if (listening) return setListening(false);
    const rec = new SR();
    rec.lang = "en-US";
    rec.onresult = (e: any) => {
      const t = e.results[0][0].transcript;
      setInput(t);
      handleAsk(t);
      setListening(false);
    };
    rec.onend = () => setListening(false);
    rec.start();
    setListening(true);
  };

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">AI Threat Assistant</h3>
          <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[10px] font-bold text-primary">LIVE</span>
        </div>
        <button onClick={() => navigate("/ai-assistant")} className="text-[10px] text-primary hover:underline">
          Open Full Chat →
        </button>
      </div>

      <div className="p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={toggleMic} className={`shrink-0 ${listening ? "border-critical text-critical animate-pulse" : ""}`}>
            {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          </Button>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk(input)}
            placeholder="Ask about threats, vulnerabilities..."
            className="flex-1 bg-secondary/50 border-border text-xs"
            disabled={loading}
          />
          <Button onClick={() => handleAsk(input)} disabled={loading || !input.trim()} size="icon" className="shrink-0">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>

        {response && (
          <div className="rounded-lg border border-primary/20 bg-secondary/30 p-4">
            <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-wrap">{response}</p>
          </div>
        )}

        {!response && !loading && (
          <p className="text-center text-[10px] text-muted-foreground py-4">Ask the AI about any cybersecurity topic</p>
        )}
      </div>
    </div>
  );
};

export default DashboardAIPanel;
