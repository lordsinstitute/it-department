"""
Lightweight ML-inspired classifier
----------------------------------
NO sklearn
NO numpy
NO external dependencies

Uses entropy + anomaly flags with a trained threshold model.
Deterministic, explainable, PyInstaller-safe.
"""

def predict(entropy, anomaly_flag):
    """
    Returns:
        1 -> Stego Likely
        0 -> Clean
    """

    score = 0

    # entropy-based signal
    if entropy > 0.90:
        score += 2
    if entropy > 0.97:
        score += 3

    # structural anomaly signal
    if anomaly_flag:
        score += 3

    # final decision threshold
    return 1 if score >= 4 else 0