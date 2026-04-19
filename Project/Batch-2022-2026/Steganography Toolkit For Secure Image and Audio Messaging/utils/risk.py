def compute_risk_label(score: int) -> str:
    score = int(score)
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"

def risk_score_encoding(
    usage_percent: float,
    bits_per_unit: int,
    encrypted: bool,
    has_password: bool,
    compressed: bool,
    password_strength: str,
):
    """
    Encoding-only risk score:
    - Higher payload usage = higher suspicion (more visible anomalies)
    - Higher bits_per_unit = more distortion and easier detection
    - Encryption + compression can increase 'maliciousness' perception
    - Weak/no password increases risk (if encrypted, weak pass is worse)
    """
    score = 0

    # Capacity usage
    if usage_percent < 5:
        score += 10
    elif usage_percent < 15:
        score += 20
    elif usage_percent < 30:
        score += 35
    elif usage_percent < 50:
        score += 55
    else:
        score += 75

    # Embedding aggressiveness
    if bits_per_unit == 1:
        score += 5
    elif bits_per_unit == 2:
        score += 15
    else:
        score += 25

    # Crypto signaling
    if encrypted:
        score += 10
        if not has_password:
            score += 20  # should not happen (encrypted implies password), but safe
        if password_strength == "weak":
            score += 15
        elif password_strength == "medium":
            score += 8
        else:
            score += 3

    if compressed:
        score += 6

    return min(100, max(1, int(score)))