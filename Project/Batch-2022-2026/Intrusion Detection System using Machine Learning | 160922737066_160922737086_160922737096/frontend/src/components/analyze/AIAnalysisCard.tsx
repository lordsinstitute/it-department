import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Brain, Loader2, Sparkles, ShieldAlert, Bug, Lightbulb, BookOpen, Volume2, VolumeX, Mic, MicOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { askAI } from "@/lib/api";

interface AIAnalysisCardProps {
  prediction: string;
  risk_score: number;
  explanation: string;
  userInput: string;
}

const AIAnalysisCard = ({ prediction, risk_score, explanation, userInput }: AIAnalysisCardProps) => {
  const [aiResponse, setAiResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const fetchAIAnalysis = async () => {
    setLoading(true);
    setAiResponse("");
    try {
      const prompt = `Analyze this network activity:\n\nInput: ${userInput.slice(0, 500)}\nPrediction: ${prediction}\nRisk Score: ${risk_score}\nExplanation: ${explanation}\n\nExplain:\n1. What type of attack this is\n2. Why it is dangerous\n3. Real-world example of this attack\n4. Step-by-step prevention measures\n\nFormat with clear headings and bullet points.`;
      const answer = await askAI(prompt);
      setAiResponse(answer);
      setHasAnalyzed(true);
    } catch (err: any) {
      setAiResponse(`⚠️ AI analysis unavailable: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!hasAnalyzed) fetchAIAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup speech on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
    };
  }, []);

  const quickPrompts = [
    { label: "Explain simply", icon: Lightbulb, prompt: `Explain this threat in simple terms a non-technical person would understand:\nPrediction: ${prediction}\nRisk Score: ${risk_score}\nExplanation: ${explanation}` },
    { label: "Prevention steps", icon: ShieldAlert, prompt: `Give me detailed step-by-step prevention measures for this specific threat:\nPrediction: ${prediction}\nRisk: ${risk_score}\nExplanation: ${explanation}` },
    { label: "Is this real?", icon: Bug, prompt: `Is this a real attack or a false positive? Analyze carefully:\nInput: ${userInput.slice(0, 300)}\nPrediction: ${prediction}\nRisk: ${risk_score}\nExplanation: ${explanation}\n\nGive your confidence level and reasoning.` },
  ];

  const handleQuickPrompt = async (prompt: string) => {
    setLoading(true);
    setAiResponse("");
    try {
      const answer = await askAI(prompt);
      setAiResponse(answer);
    } catch (err: any) {
      setAiResponse(`⚠️ AI unavailable: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Speaker: Read aloud ──
  const toggleSpeak = () => {
    if (!("speechSynthesis" in window)) return;

    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }

    if (!aiResponse) return;

    // Strip markdown formatting for cleaner speech
    const cleanText = aiResponse
      .replace(/\*\*/g, "")
      .replace(/###?\s?/g, "")
      .replace(/[-•]\s/g, "")
      .replace(/\n+/g, ". ");

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  };

  // ── Mic: Voice question ──
  const toggleMic = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;

    if (listening) {
      setListening(false);
      return;
    }

    const rec = new SR();
    rec.lang = "en-US";
    rec.onresult = async (e: any) => {
      const transcript = e.results[0][0].transcript;
      setListening(false);
      // Use voice input as a question about the analysis
      setLoading(true);
      setAiResponse("");
      try {
        const prompt = `Context:\nPrediction: ${prediction}\nRisk: ${risk_score}\nExplanation: ${explanation}\n\nUser asks by voice: "${transcript}"\n\nAnswer concisely and helpfully.`;
        const answer = await askAI(prompt);
        setAiResponse(answer);
      } catch (err: any) {
        setAiResponse(`⚠️ ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    rec.onerror = () => setListening(false);
    rec.onend = () => setListening(false);
    rec.start();
    setListening(true);
  };

  const hasSpeech = typeof window !== "undefined" && "speechSynthesis" in window;
  const hasMic = typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  const formatResponse = (text: string) => {
    return text.split("\n").map((line, i) => {
      const trimmed = line.trim();
      if (trimmed.startsWith("###")) {
        return <p key={i} className="text-xs font-bold text-primary mt-3 mb-1">{trimmed.replace(/^#+\s*/, "")}</p>;
      }
      if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
        return <p key={i} className="text-xs font-bold text-foreground mt-3 mb-1">{trimmed.replace(/\*\*/g, "")}</p>;
      }
      if (/^\*\s?\*\*/.test(trimmed)) {
        const clean = trimmed.replace(/^\*\s?\*\*/, "").replace(/\*\*:?/, ":");
        return <p key={i} className="text-xs text-foreground/80 pl-3 leading-relaxed">• <strong>{clean.split(":")[0]}</strong>:{clean.split(":").slice(1).join(":")}</p>;
      }
      if (trimmed.startsWith("- ") || trimmed.startsWith("• ") || trimmed.startsWith("* ")) {
        return <p key={i} className="text-xs text-foreground/80 pl-3 leading-relaxed">• {trimmed.slice(2)}</p>;
      }
      if (/^\d+\.\s/.test(trimmed)) {
        return <p key={i} className="text-xs text-foreground/80 pl-3 leading-relaxed">{trimmed}</p>;
      }
      if (!trimmed) return <div key={i} className="h-1.5" />;
      return <p key={i} className="text-xs text-foreground/80 leading-relaxed">{trimmed}</p>;
    });
  };

  return (
    <motion.div
      className="glass-card rounded-xl p-5 border border-primary/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-primary/10 p-1.5">
            <Sparkles className="h-4 w-4 text-primary" />
          </div>
          <h3 className="text-sm font-semibold text-foreground">AI Threat Intelligence</h3>
          <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[9px] font-bold text-primary">AI POWERED</span>
        </div>

        {/* Speaker & Mic controls */}
        <div className="flex items-center gap-1.5">
          {hasMic && (
            <Button
              variant="outline"
              size="icon"
              onClick={toggleMic}
              disabled={loading}
              className={`h-7 w-7 ${listening ? "border-critical text-critical animate-pulse" : "border-border text-muted-foreground"}`}
              title="Ask by voice"
            >
              {listening ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
            </Button>
          )}
          {hasSpeech && (
            <Button
              variant="outline"
              size="icon"
              onClick={toggleSpeak}
              disabled={loading || !aiResponse}
              className={`h-7 w-7 ${speaking ? "border-primary text-primary" : "border-border text-muted-foreground"}`}
              title={speaking ? "Stop reading" : "Read aloud"}
            >
              {speaking ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
            </Button>
          )}
        </div>
      </div>

      {/* Listening indicator */}
      {listening && (
        <div className="flex items-center gap-2 mb-3 rounded-lg bg-critical/10 border border-critical/20 px-3 py-2">
          <Mic className="h-3.5 w-3.5 text-critical animate-pulse" />
          <span className="text-[11px] text-critical font-medium">Listening... speak your question</span>
        </div>
      )}

      {/* Response area */}
      <div className="rounded-lg bg-secondary/30 border border-border p-4 max-h-[300px] overflow-y-auto">
        {loading ? (
          <div className="flex items-center gap-3 py-6 justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <div>
              <p className="text-xs font-medium text-foreground">AI is thinking...</p>
              <p className="text-[10px] text-muted-foreground">Analyzing threat patterns</p>
            </div>
          </div>
        ) : aiResponse ? (
          <div>{formatResponse(aiResponse)}</div>
        ) : null}
      </div>

      {/* Quick action buttons */}
      <div className="flex flex-wrap gap-2 mt-3">
        {quickPrompts.map((qp) => (
          <Button
            key={qp.label}
            variant="outline"
            size="sm"
            disabled={loading}
            onClick={() => handleQuickPrompt(qp.prompt)}
            className="text-[10px] h-7 border-primary/30 text-primary hover:bg-primary/10 gap-1.5"
          >
            <qp.icon className="h-3 w-3" />
            {qp.label}
          </Button>
        ))}
        <Button
          variant="outline"
          size="sm"
          disabled={loading}
          onClick={fetchAIAnalysis}
          className="text-[10px] h-7 border-border text-muted-foreground hover:bg-secondary gap-1.5"
        >
          <BookOpen className="h-3 w-3" />
          Re-analyze
        </Button>
      </div>
    </motion.div>
  );
};

export default AIAnalysisCard;
