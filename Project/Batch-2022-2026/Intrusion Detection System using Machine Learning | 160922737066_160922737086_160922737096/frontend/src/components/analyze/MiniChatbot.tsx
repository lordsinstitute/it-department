import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, Send, Loader2, User, Trash2, X, MessageSquare, Mic, MicOff } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { askAI } from "@/lib/api";

interface ChatMessage {
  id: string;
  role: "user" | "ai";
  content: string;
}

interface MiniChatbotProps {
  context?: { prediction: string; risk_score: number; explanation: string; userInput: string } | null;
}

const MiniChatbot = ({ context }: MiniChatbotProps) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      let prompt = text.trim();
      if (context) {
        prompt = `Context from latest analysis:\nPrediction: ${context.prediction}\nRisk Score: ${context.risk_score}\nExplanation: ${context.explanation}\n\nUser question: ${text.trim()}\n\nAnswer based on the analysis context above. Be concise and helpful.`;
      }
      const answer = await askAI(prompt);
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "ai", content: answer }]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "ai", content: `⚠️ ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const toggleMic = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    if (listening) { setListening(false); return; }
    const rec = new SR();
    rec.lang = "en-US";
    rec.onresult = (e: any) => {
      const t = e.results[0][0].transcript;
      setInput(t);
      sendMessage(t);
      setListening(false);
    };
    rec.onerror = () => setListening(false);
    rec.onend = () => setListening(false);
    rec.start();
    setListening(true);
  };

  const hasSpeech = typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  if (!open) {
    return (
      <motion.button
        onClick={() => setOpen(true)}
        className="fixed bottom-4 right-4 z-[60] flex items-center gap-2 rounded-full bg-primary px-3.5 py-2.5 text-primary-foreground shadow-lg hover:shadow-xl hover:scale-105 transition-all glow-primary"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.95 }}
      >
        <MessageSquare className="h-4 w-4" />
        <span className="text-[11px] font-bold hidden sm:inline">Ask AI</span>
      </motion.button>
    );
  }

  return (
    <motion.div
      className="fixed bottom-4 right-4 z-[60] w-[calc(100vw-2rem)] sm:w-[370px] max-w-[370px] rounded-2xl border border-border bg-card shadow-2xl flex flex-col"
      style={{ maxHeight: "min(460px, calc(100dvh - 100px))" }}
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-primary/10 p-1.5">
            <Bot className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-xs font-bold text-foreground">AI Assistant</p>
            <p className="text-[9px] text-muted-foreground">
              {context ? "Context-aware mode" : "General mode"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <button onClick={() => setMessages([])} className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-secondary">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
          <button onClick={() => setOpen(false)} className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-secondary">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-2.5">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="h-10 w-10 text-muted-foreground/20 mb-2" />
            <p className="text-[11px] text-muted-foreground">Ask about the current analysis</p>
            <div className="flex flex-wrap gap-1.5 mt-3 justify-center">
              {["Explain this alert", "How to fix?", "Is this dangerous?"].map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="rounded-full border border-border bg-secondary/30 px-2.5 py-1 text-[9px] text-muted-foreground hover:bg-secondary/60 hover:text-foreground transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "ai" && (
                <div className="shrink-0 rounded-full bg-primary/10 p-1.5 h-6 w-6 flex items-center justify-center mt-0.5">
                  <Bot className="h-3 w-3 text-primary" />
                </div>
              )}
              <div className={`max-w-[80%] rounded-xl px-3 py-2 text-[11px] leading-relaxed ${
                msg.role === "user"
                  ? "bg-primary text-primary-foreground rounded-br-sm"
                  : "bg-secondary/50 border border-border text-foreground rounded-bl-sm"
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
              {msg.role === "user" && (
                <div className="shrink-0 rounded-full bg-secondary p-1.5 h-6 w-6 flex items-center justify-center mt-0.5">
                  <User className="h-3 w-3 text-muted-foreground" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        {loading && (
          <div className="flex gap-2">
            <div className="shrink-0 rounded-full bg-primary/10 p-1.5 h-6 w-6 flex items-center justify-center">
              <Bot className="h-3 w-3 text-primary" />
            </div>
            <div className="bg-secondary/50 border border-border rounded-xl rounded-bl-sm px-3 py-2">
              <div className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin text-primary" />
                <span className="text-[10px] text-muted-foreground">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-border px-3 py-2.5 flex items-center gap-2">
        {hasSpeech && (
          <Button
            variant="outline"
            size="icon"
            onClick={toggleMic}
            className={`h-8 w-8 shrink-0 ${listening ? "border-critical text-critical animate-pulse" : ""}`}
          >
            {listening ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
          </Button>
        )}
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
          placeholder="Ask about this threat..."
          className="flex-1 bg-secondary/50 border-border text-xs h-8"
          disabled={loading}
        />
        <Button onClick={() => sendMessage(input)} disabled={loading || !input.trim()} size="icon" className="h-8 w-8 shrink-0">
          <Send className="h-3.5 w-3.5" />
        </Button>
      </div>
    </motion.div>
  );
};

export default MiniChatbot;
