def compute_risk(payload: dict):
    """
    Simple, explainable risk model:
    - Score based on open ports + risky ports
    - Outputs: (level, score, reasons[])
    """
    results = payload.get("results") or []
    open_total = 0
    risky_hits = 0
    reasons = []

    risky_ports = {21, 23, 445, 3389, 5900, 1433, 3306, 6379, 27017}
    medium_ports = {22, 80, 8080, 8443, 53, 110, 143}

    for t in results:
        for op in (t.get("open_ports") or []):
            open_total += 1
            p = int(op.get("port", 0))
            if p in risky_ports:
                risky_hits += 1

    score = 0
    score += open_total * 2
    score += risky_hits * 15

    # Bonus if many ports open
    if open_total >= 10:
        score += 15
        reasons.append("Many open ports detected (>=10).")

    if risky_hits > 0:
        reasons.append(f"Risky ports detected: {risky_hits} hit(s).")

    # Level mapping
    if score >= 70:
        level = "Critical"
    elif score >= 40:
        level = "High"
    elif score >= 15:
        level = "Medium"
    else:
        level = "Low"

    if not reasons:
        reasons.append("No high-risk indicators triggered by the scoring model.")

    return level, int(score), reasons