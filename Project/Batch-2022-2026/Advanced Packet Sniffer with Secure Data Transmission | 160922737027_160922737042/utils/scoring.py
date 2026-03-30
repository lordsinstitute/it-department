def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def risk_label_from_score(score: int) -> str:
    """
    Risk bands:
    0-24: Low
    25-49: Medium
    50-74: High
    75-100: Critical
    """
    s = clamp(score, 0, 100)
    if s >= 75:
        return "Critical"
    if s >= 50:
        return "High"
    if s >= 25:
        return "Medium"
    return "Low"