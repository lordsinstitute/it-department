import re
from typing import Dict, Any

from .ml_engine import load_model_artifacts

SUSPICIOUS_KEYWORDS = {
    "urgent", "immediately", "verify", "password", "otp", "bank", "gift card",
    "wire", "refund", "lottery", "prize", "crypto", "wallet", "kyc",
    "inheritance", "claim", "bonus", "suspended", "action required",
    "final warning", "click here", "free", "reward", "limited offer",
    "confirm identity", "unusual login", "reset now", "payment failed"
}

FINANCIAL_WORDS = {
    "bank", "account", "payment", "refund", "wallet", "card", "otp",
    "wire", "transfer", "crypto", "bonus", "salary", "invoice", "upi"
}

ATTACHMENT_WORDS = {
    "attachment", "attached", "document", "file", "invoice", "pdf", "zip",
    "statement", "report", "spreadsheet"
}

URGENT_WORDS = {
    "urgent", "immediately", "now", "today", "asap", "final warning", "action required"
}

IMPERSONATION_WORDS = {
    "ceo", "hr", "finance team", "it admin", "admin", "microsoft support",
    "bank officer", "payroll team", "security team"
}

LINK_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+\b")


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_keyword_hits(text: str, word_set: set[str]) -> int:
    lower_text = text.lower()
    return sum(1 for word in word_set if word in lower_text)


def extract_features(text: str) -> Dict[str, Any]:
    lower_text = text.lower()
    alpha_count = max(1, sum(1 for c in text if c.isalpha()))
    upper_count = sum(1 for c in text if c.isupper())

    return {
        "suspicious_links": len(LINK_PATTERN.findall(lower_text)),
        "email_count": len(EMAIL_PATTERN.findall(lower_text)),
        "urgent_count": count_keyword_hits(lower_text, URGENT_WORDS),
        "financial_count": count_keyword_hits(lower_text, FINANCIAL_WORDS),
        "attachment_count": count_keyword_hits(lower_text, ATTACHMENT_WORDS),
        "impersonation_count": count_keyword_hits(lower_text, IMPERSONATION_WORDS),
        "suspicious_keyword_count": count_keyword_hits(lower_text, SUSPICIOUS_KEYWORDS),
        "exclamations": lower_text.count("!"),
        "uppercase_ratio": upper_count / alpha_count,
        "has_otp": 1 if "otp" in lower_text else 0,
        "has_password": 1 if "password" in lower_text else 0,
        "has_verify": 1 if "verify" in lower_text or "verification" in lower_text else 0,
    }


def compute_risk(features: Dict[str, Any], ml_label: str, confidence: float) -> tuple[int, str]:
    risk = 0

    risk += features["suspicious_links"] * 16
    risk += features["urgent_count"] * 8
    risk += features["financial_count"] * 12
    risk += features["attachment_count"] * 4
    risk += features["impersonation_count"] * 10
    risk += features["suspicious_keyword_count"] * 6
    risk += min(features["exclamations"] * 2, 10)

    if features["uppercase_ratio"] > 0.35:
        risk += 10

    if features["has_otp"]:
        risk += 14
    if features["has_password"]:
        risk += 12
    if features["has_verify"]:
        risk += 8

    if ml_label == "spam":
        risk += int(18 + confidence * 18)
    elif ml_label == "scam":
        risk += int(32 + confidence * 22)
    else:
        risk += int(confidence * 5)

    risk = min(risk, 100)

    if risk >= 85:
        level = "Critical"
    elif risk >= 65:
        level = "High"
    elif risk >= 35:
        level = "Medium"
    else:
        level = "Low"

    return risk, level


def final_label(ml_label: str, risk_level: str, features: Dict[str, Any]) -> str:
    if ml_label == "scam":
        return "Scam"

    if ml_label == "spam":
        if features["financial_count"] >= 1 or features["impersonation_count"] >= 1:
            return "Scam"
        return "Spam"

    if risk_level in ("High", "Critical"):
        if features["financial_count"] >= 1 or features["has_otp"] or features["has_password"]:
            return "Scam"
        return "Spam"

    return "Safe"


def build_summary(label: str, ml_label: str, risk_level: str, risk_score: int, features: Dict[str, Any]) -> str:
    return (
        f"ML model predicted '{ml_label}'. Final classification is {label}. "
        f"Risk level is {risk_level} with score {risk_score}/100. "
        f"Detected {features['suspicious_links']} suspicious link(s), "
        f"{features['urgent_count']} urgency indicator(s), "
        f"{features['financial_count']} financial keyword(s), "
        f"{features['impersonation_count']} impersonation indicator(s), and "
        f"{features['attachment_count']} attachment-related term(s)."
    )


def build_finding_details(features: Dict[str, Any], probabilities: Dict[str, float]) -> str:
    return (
        f"ML probabilities -> safe: {probabilities.get('safe', 0):.2%}, "
        f"spam: {probabilities.get('spam', 0):.2%}, "
        f"scam: {probabilities.get('scam', 0):.2%} | "
        f"Suspicious links: {features['suspicious_links']} | "
        f"Urgency indicators: {features['urgent_count']} | "
        f"Financial keywords: {features['financial_count']} | "
        f"Impersonation indicators: {features['impersonation_count']} | "
        f"Attachment keywords: {features['attachment_count']} | "
        f"Embedded email addresses: {features['email_count']} | "
        f"Uppercase ratio: {features['uppercase_ratio']:.2f}"
    )


def analyze_email_content(text: str) -> Dict[str, Any]:
    clean = clean_text(text)
    if not clean:
        return {
            "cleaned_text": "",
            "ml_label": "safe",
            "prediction_label": "Safe",
            "confidence_score": 0.0,
            "risk_score": 0,
            "risk_level": "Low",
            "suspicious_links": 0,
            "urgent_words": 0,
            "financial_words": 0,
            "attachment_words": 0,
            "impersonation_words": 0,
            "summary": "No text provided for analysis.",
            "finding_details": "Empty input"
        }

    model, vectorizer, encoder = load_model_artifacts()
    X = vectorizer.transform([clean])

    predicted_index = model.predict(X)[0]
    predicted_label = str(encoder.inverse_transform([predicted_index])[0])

    probabilities_raw = model.predict_proba(X)[0]
    label_names = list(encoder.classes_)
    probabilities = {label_names[i]: float(probabilities_raw[i]) for i in range(len(label_names))}
    confidence = max(probabilities.values())

    features = extract_features(clean)
    risk_score, risk_level = compute_risk(features, predicted_label, confidence)
    label = final_label(predicted_label, risk_level, features)

    return {
        "cleaned_text": clean,
        "ml_label": predicted_label,
        "prediction_label": label,
        "confidence_score": round(confidence * 100, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "suspicious_links": features["suspicious_links"],
        "urgent_words": features["urgent_count"],
        "financial_words": features["financial_count"],
        "attachment_words": features["attachment_count"],
        "impersonation_words": features["impersonation_count"],
        "summary": build_summary(label, predicted_label, risk_level, risk_score, features),
        "finding_details": build_finding_details(features, probabilities)
    }