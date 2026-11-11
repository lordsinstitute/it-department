def calculate_risk(result):
    score = result.get("risk_score", 0)

    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"