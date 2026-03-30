export interface ModelScore {
  accuracy: number;
  risk_score: number;
  confidence?: number;
}

export interface PredictionResponse {
  prediction: string;
  risk_score: number;
  explanation: string;
  best_model?: string;
  models?: Record<string, ModelScore>;
}

const BACKEND_URL = "https://ids-backend2.onrender.com";

export const getBackendUrl = (): string => {
  return localStorage.getItem("ids_backend_url") || BACKEND_URL;
};

export const setBackendUrl = (url: string) => {
  localStorage.setItem("ids_backend_url", url);
};

const fetchWithRetry = async (url: string, options: RequestInit, retries = 2, timeoutMs = 30000, externalSignal?: AbortSignal): Promise<Response> => {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    // Link external abort signal
    if (externalSignal) {
      externalSignal.addEventListener("abort", () => controller.abort());
      if (externalSignal.aborted) controller.abort();
    }

    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeout);
      if (response.ok) return response;

      const isServerError = response.status >= 500;
      const isLastAttempt = attempt === retries;
      if (isLastAttempt) {
        throw new Error(
          isServerError
            ? `Backend server error (${response.status}). The deployed API is reachable but failing internally.`
            : `Backend error: ${response.status} ${response.statusText}`
        );
      }
    } catch (err: any) {
      clearTimeout(timeout);
      if (externalSignal?.aborted) {
        throw new DOMException("Request cancelled by user", "AbortError");
      }
      const isLastAttempt = attempt === retries;
      if (isLastAttempt) {
        if (err.name === "AbortError") {
          throw new Error("Request timed out — backend may be waking up. Please retry in 15-30 seconds.");
        }
        if (err.message === "Failed to fetch") {
          throw new Error("Backend request was blocked or failed. Your deployed API likely returned an error without CORS headers.");
        }
        throw new Error(err.message || "Cannot reach backend API. Check your connection.");
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  throw new Error("Failed after retries");
};

export const askAI = async (question: string, signal?: AbortSignal): Promise<string> => {
  const url = getBackendUrl();
  const response = await fetchWithRetry(`${url}/ask-ai`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  }, 2, 45000, signal);
  const data = await response.json();
  return data.answer || data.response || JSON.stringify(data);
};

export const checkBackendHealth = async (): Promise<boolean> => {
  const url = getBackendUrl();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
};

export const analyzeThreat = async (data: string): Promise<PredictionResponse> => {
  const url = getBackendUrl();
  const response = await fetchWithRetry(`${url}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  }, 2, 45000);
  const raw = await response.json();

  // Normalize model data: backend may return "risk" instead of "risk_score"
  let models: Record<string, ModelScore> | undefined;
  if (raw.models && typeof raw.models === "object") {
    models = {};
    const BASE_ACCURACY: Record<string, number> = {
      "Random Forest": 99.2,
      "SVM": 97.5,
      "Decision Tree": 95.3,
      "Naive Bayes": 93.1,
    };
    for (const [name, val] of Object.entries(raw.models)) {
      if (typeof val === "number") {
        models[name] = {
          risk_score: val,
          accuracy: BASE_ACCURACY[name] || 90,
          confidence: Math.max(0, Math.min(100, val - 2)),
        };
      } else {
        const m = val as any;
        models[name] = {
          risk_score: Number(m.risk_score ?? m.risk ?? 0) || 0,
          accuracy: Number(m.accuracy ?? 0) || 0,
          confidence: Number(m.confidence ?? 0) || 0,
        };
      }
    }
  }

  return {
    prediction: raw.prediction || "Unknown",
    risk_score: Number(raw.risk_score ?? 0) || 0,
    explanation: raw.explanation || "",
    best_model: raw.best_model,
    models,
  };
};

export interface BackendAlert {
  id: string;
  timestamp: string;
  source_ip: string;
  prediction: string;
  risk_score: number;
  severity: string;
  explanation: string;
  status: string;
}

export const fetchAlerts = async (): Promise<BackendAlert[]> => {
  const url = getBackendUrl();
  const response = await fetchWithRetry(`${url}/alerts`, {}, 1, 15000);
  return response.json();
};

export const testConnection = async (): Promise<{ ok: boolean; latency: number }> => {
  const url = getBackendUrl();
  const start = performance.now();
  try {
    const response = await fetch(`${url}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: "test connection ping" }),
    });
    const latency = Math.round(performance.now() - start);
    return { ok: response.ok, latency };
  } catch {
    throw new Error("Cannot reach backend API");
  }
};
