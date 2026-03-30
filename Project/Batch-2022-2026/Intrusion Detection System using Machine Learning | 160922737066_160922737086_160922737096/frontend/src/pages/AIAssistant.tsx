import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Send, Loader2, Bot, User, Mic, MicOff, Volume2, VolumeX,
  Square, Trash2, RotateCcw, Zap
} from "lucide-react";
import { supabase } from "@/lib/supabase";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/context/AuthContext";
import { askAI, checkBackendHealth } from "@/lib/api";
import ReactMarkdown from "react-markdown";

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
  timestamp: Date;
  isError?: boolean;
}

const QUICK_SUGGESTIONS = [
  "What is SQL injection?",
  "How to prevent DDoS attacks?",
  "Explain XSS vulnerabilities",
  "What is a brute force attack?",
];

const TypingText = ({
  text,
  animate,
  onComplete,
}: {
  text: string;
  animate: boolean;
  onComplete?: () => void;
}) => {
  const [displayed, setDisplayed] = useState(animate ? "" : text);
  const [done, setDone] = useState(!animate);

  useEffect(() => {
    if (!animate) {
      setDisplayed(text);
      setDone(true);
      return;
    }

    if (text.length <= 100) {
      setDisplayed(text);
      setDone(true);
      onComplete?.();
      return;
    }

    setDisplayed("");
    setDone(false);
    let i = 0;
    let cancelled = false;
    const speed = Math.max(5, Math.min(20, 2000 / text.length));
    const interval = setInterval(() => {
      if (cancelled) return;
      i += 3;
      if (i >= text.length) {
        setDisplayed(text);
        setDone(true);
        onComplete?.();
        clearInterval(interval);
      } else {
        setDisplayed(text.slice(0, i));
      }
    }, speed);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [text, animate, onComplete]);

  return (
    <div className="prose prose-sm prose-invert max-w-none text-sm leading-relaxed [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_h1]:text-base [&_h2]:text-sm [&_h3]:text-xs [&_code]:text-primary [&_code]:bg-primary/10 [&_code]:px-1 [&_code]:rounded">
      <ReactMarkdown>{displayed}</ReactMarkdown>
      {!done && <span className="inline-block h-4 w-1.5 animate-pulse align-middle bg-primary ml-0.5" />}
    </div>
  );
};

const AIAssistant = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [speakEnabled, setSpeakEnabled] = useState(true);
  const [backendStatus, setBackendStatus] = useState<"unknown" | "online" | "waking">("unknown");
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const { toast } = useToast();
  const { user } = useAuth();

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  // Cancel any leftover speech & health check on mount
  useEffect(() => {
    window.speechSynthesis?.cancel();
    checkBackendHealth().then((ok) => {
      setBackendStatus(ok ? "online" : "waking");
      if (!ok) {
        toast({ title: "Server is waking up", description: "First request may take 15-30 seconds.", variant: "default" });
      }
    });
    return () => {
      window.speechSynthesis?.cancel();
    };
  }, []);

  const speak = (text: string) => {
    if (!speakEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const clean = text.replace(/\*\*/g, "").replace(/###?\s?/g, "").replace(/[-•]\s/g, "").replace(/\n+/g, ". ");
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  const toggleSpeak = () => {
    if (speakEnabled) {
      window.speechSynthesis?.cancel();
      setIsSpeaking(false);
    }
    setSpeakEnabled((prev) => !prev);
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
    setTypingMessageId(null);
    setLoading(false);
  };

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;
    const question = text.trim();
    setLastQuestion(question);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: question, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const answer = await askAI(question, controller.signal);
      setBackendStatus("online");

      const aiMsg: Message = { id: crypto.randomUUID(), role: "ai", content: answer, timestamp: new Date() };
      setTypingMessageId(aiMsg.id);
      setMessages((prev) => [...prev, aiMsg]);
      speak(answer);

      // Save silently
      supabase.from("ai_logs").insert({
        question,
        answer,
        timestamp: new Date().toISOString(),
        user_id: user?.id,
      }).then(() => {});
    } catch (err: any) {
      if (err.name === "AbortError" || err.message?.includes("cancelled")) {
        setMessages((prev) => [...prev, {
          id: crypto.randomUUID(), role: "ai",
          content: "⏹️ Response stopped.", timestamp: new Date(),
        }]);
      } else {
        setMessages((prev) => [...prev, {
          id: crypto.randomUUID(), role: "ai", isError: true,
          content: `⚠️ Backend not responding: ${err.message}\n\nClick **Retry** below to try again.`,
          timestamp: new Date(),
        }]);
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }, [loading, user, speakEnabled]);

  const retryLast = () => {
    if (lastQuestion) sendMessage(lastQuestion);
  };

  const clearChat = () => {
    setMessages([]);
    setLastQuestion(null);
    window.speechSynthesis?.cancel();
    toast({ title: "Chat cleared" });
  };

  const toggleMic = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast({ title: "Speech not supported", variant: "destructive" });
      return;
    }
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.onresult = (e: any) => {
      const transcript = e.results[0][0].transcript;
      setInput(transcript);
      sendMessage(transcript);
      setListening(false);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  };

  const hasError = messages.length > 0 && messages[messages.length - 1]?.isError;

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] sm:h-[calc(100vh-120px)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 sm:mb-4 gap-2">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="rounded-lg bg-primary/10 p-1.5 sm:p-2 shrink-0">
            <Bot className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
          </div>
          <div className="min-w-0">
            <h2 className="text-base sm:text-lg font-bold text-foreground truncate">AI Threat Assistant</h2>
            <div className="flex items-center gap-2">
              <p className="text-[10px] text-muted-foreground hidden sm:inline">Powered by IDS Backend AI</p>
              <span className={`inline-block h-1.5 w-1.5 rounded-full ${
                backendStatus === "online" ? "bg-primary" :
                backendStatus === "waking" ? "bg-accent-foreground animate-pulse" : "bg-muted-foreground"
              }`} />
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearChat} className="text-muted-foreground h-8 gap-1 text-xs">
              <Trash2 className="h-3.5 w-3.5" /> Clear
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={toggleSpeak} className="text-muted-foreground h-8">
            {speakEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* Backend waking banner */}
      {backendStatus === "waking" && (
        <div className="rounded-lg bg-accent/50 border border-accent px-3 py-2 mb-3 flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-accent-foreground" />
          <span className="text-xs text-accent-foreground">Server is waking up — first request may take 15-30 seconds...</span>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-3 pr-1 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="h-16 w-16 text-muted-foreground/20 mb-4" />
            <p className="text-sm text-muted-foreground">Ask me anything about cybersecurity threats</p>
            <p className="text-[10px] text-muted-foreground mt-2 mb-4">Type, use the mic 🎤, or try a suggestion below</p>

            {/* Quick suggestions */}
            <div className="flex flex-wrap justify-center gap-1.5 sm:gap-2 max-w-md px-2">
              {QUICK_SUGGESTIONS.map((q) => (
                <Button
                  key={q}
                  variant="outline"
                  size="sm"
                  onClick={() => sendMessage(q)}
                  className="text-[10px] sm:text-[11px] h-7 sm:h-8 border-primary/30 text-primary hover:bg-primary/10 gap-1 sm:gap-1.5 px-2 sm:px-3"
                >
                  <Zap className="h-3 w-3 hidden sm:inline" />
                  {q}
                </Button>
              ))}
            </div>
          </div>
        )}
        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "ai" && (
                <div className="shrink-0 rounded-full bg-primary/10 p-2 h-8 w-8 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground rounded-br-md"
                    : msg.isError
                    ? "bg-destructive/10 border border-destructive/30 rounded-bl-md"
                    : "glass-card border border-border rounded-bl-md"
                }`}
              >
                {msg.role === "ai" && !msg.isError ? (
                  <TypingText
                    text={msg.content}
                    animate={msg.id === typingMessageId}
                    onComplete={() => {
                      if (typingMessageId === msg.id) setTypingMessageId(null);
                    }}
                  />
                ) : (
                  <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                )}
                <p className="text-[9px] mt-1 opacity-50">
                  {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </p>
              </div>
              {msg.role === "user" && (
                <div className="shrink-0 rounded-full bg-secondary p-2 h-8 w-8 flex items-center justify-center">
                  <User className="h-4 w-4 text-muted-foreground" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
            <div className="shrink-0 rounded-full bg-primary/10 p-2 h-8 w-8 flex items-center justify-center">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="glass-card rounded-2xl rounded-bl-md px-4 py-3 border border-border flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <span className="text-xs text-muted-foreground">Thinking...</span>
            </div>
          </motion.div>
        )}
      </div>

      {/* Input */}
      <div className="space-y-2">
        {(loading || typingMessageId !== null || isSpeaking) && (
          <div className="flex justify-center">
            <Button
              onClick={stopGeneration}
              variant="outline"
              className="rounded-full h-10 px-5 gap-2 border-border bg-secondary text-foreground hover:bg-muted"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              <span className="text-xs font-medium">Stop</span>
            </Button>
          </div>
        )}

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={toggleMic}
            className={`shrink-0 ${listening ? "border-destructive text-destructive animate-pulse" : "border-border text-muted-foreground"}`}
          >
            {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          </Button>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder="Ask about threats, security, or IDS..."
            className="flex-1 bg-secondary/50 border-border"
            disabled={loading}
          />
          <Button
            onClick={() => sendMessage(input)}
            disabled={loading || !input.trim()}
            size="icon"
            className="shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;
