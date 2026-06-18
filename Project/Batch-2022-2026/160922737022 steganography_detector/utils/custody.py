import json, hashlib, datetime

def generate_custody_record(scan):
    record = {
        "file": scan.filename,
        "timestamp": scan.timestamp.isoformat(),
        "risk": scan.risk,
        "hash": scan.hash_value,
        "analyst": "System",
        "tool": "Steganography Detector v1.0"
    }

    record["signature"] = hashlib.sha256(
        json.dumps(record).encode()
    ).hexdigest()

    path = f"custody_{scan.id}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return path