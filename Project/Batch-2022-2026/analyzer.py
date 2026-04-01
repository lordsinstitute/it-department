import json
import os
import re
from datetime import datetime
from utils.helpers import safe_read_text, load_json_file
from config import Config

# Local "AI-driven TI" simulation:
# - weights.json = simple word/phrase weights
# - threat_feed.json = local threat intel feed (IOCs, actor tags, TTP keywords)

SUSPICIOUS_PATTERNS = [
    (r"\bCVE-\d{4}-\d{4,7}\b", 12, "CVE mentioned"),
    (r"\b(base64|frombase64string)\b", 10, "Base64 usage"),
    (r"\b(powershell|cmd\.exe|wscript|cscript)\b", 10, "Windows scripting"),
    (r"\b(mimikatz|bloodhound|cobalt\s*strike)\b", 15, "Known offensive tooling"),
    (r"\b(exfiltrat|data\s*leak|ransom|encrypt\s+files)\b", 14, "Exfiltration/ransom signals"),
    (r"\b(credential|password\s*dump|lsass)\b", 12, "Credential access signals"),
    (r"\b(reverse\s*shell|meterpreter|beacon)\b", 15, "C2 / reverse shell"),
    (r"\b(\/admin|sql\s*injection|xss|csrf)\b", 8, "Web attack terms"),
    (r"\b(0day|zero\s*day)\b", 12, "Zero-day mention"),
]

def analyze_input(text: str) -> dict:
    """
    Returns a dict with:
      - score, level
      - findings: list
      - intel_hits: list
      - indicators: counts
      - summary
    Never raises (caller expects demo-safe).
    """
    try:
        text = (text or "").strip()
        lowered = text.lower()

        weights = load_json_file(os.path.join(Config.ROOT_DIR, "models", "weights.json"), default={})
        feed = load_json_file(os.path.join(Config.ROOT_DIR, "models", "threat_feed.json"), default={})

        findings = []
        intel_hits = []
        indicators = {"patterns": 0, "ioc_hits": 0, "weighted_hits": 0, "length": len(text)}

        score = 0

        # Pattern-based signals
        for pattern, pts, label in SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                score += pts
                indicators["patterns"] += 1
                findings.append({"type": "pattern", "label": label, "points": pts, "pattern": pattern})

        # Weighted "AI" words/phrases
        # Simple: add points for any keyword occurrences
        for key, pts in weights.get("keywords", {}).items():
            if key.lower() in lowered:
                score += int(pts)
                indicators["weighted_hits"] += 1
                findings.append({"type": "keyword", "label": f"Keyword: {key}", "points": int(pts)})

        # Threat intel feed matching (IOCs + TTP keywords)
        # This simulates local TI correlation
        for ioc in feed.get("iocs", []):
            if ioc.lower() in lowered:
                score += 18
                indicators["ioc_hits"] += 1
                intel_hits.append({"kind": "IOC", "value": ioc, "points": 18})

        for ttp in feed.get("ttp_keywords", []):
            if ttp.lower() in lowered:
                score += 10
                intel_hits.append({"kind": "TTP", "value": ttp, "points": 10})

        # Length heuristic (very long suspicious dumps often include encoded blobs/logs)
        if len(text) > 1500:
            score += 6
            findings.append({"type": "heuristic", "label": "Large content size", "points": 6})

        # Normalize / cap to 100
        score = min(100, max(0, score))

        level = risk_level(score)

        summary = build_summary(score, level, indicators, findings, intel_hits)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "score": score,
            "level": level,
            "indicators": indicators,
            "findings": findings,
            "intel_hits": intel_hits,
            "summary": summary,
        }
    except Exception as e:
        # Demo-safe fallback
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "score": 0,
            "level": "Low",
            "indicators": {"patterns": 0, "ioc_hits": 0, "weighted_hits": 0, "length": 0},
            "findings": [],
            "intel_hits": [],
            "summary": f"Analyzer failed safely: {type(e).__name__}. No threat inferred.",
        }

def risk_level(score: int) -> str:
    if score >= 80:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"

def build_summary(score, level, indicators, findings, intel_hits) -> str:
    key = []
    if indicators.get("ioc_hits", 0) > 0:
        key.append("threat-intel IOC match")
    if indicators.get("patterns", 0) > 0:
        key.append("attack-pattern signals")
    if indicators.get("weighted_hits", 0) > 0:
        key.append("keyword risk indicators")

    if not key:
        key_text = "no strong indicators detected"
    else:
        key_text = ", ".join(key)

    return f"Predicted risk is {level} (score {score}/100) with {key_text}."

def extract_text_from_upload(file_path: str) -> str:
    # Keep minimal: treat as plain text. If binary, safe_read_text will handle.
    return safe_read_text(file_path, max_chars=20000)