import hashlib
import json
from datetime import datetime
from detector.config import Config

LEDGER_PATH = Config.DATABASE_DIR / "evidence_ledger.json"


def compute_scan_hash(scan, previous_hash: str) -> str:
    payload = {
        "scan_id": scan.id,
        "user_id": scan.user_id,
        "source_type": scan.source_type,
        "original_filename": scan.original_filename,
        "ml_label": scan.ml_label,
        "prediction_label": scan.prediction_label,
        "confidence_score": scan.confidence_score,
        "risk_score": scan.risk_score,
        "risk_level": scan.risk_level,
        "summary": scan.summary,
        "finding_details": scan.finding_details,
        "created_at": scan.created_at.isoformat() if scan.created_at else "",
        "previous_hash": previous_hash
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_ledger():
    if not LEDGER_PATH.exists():
        return []
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def append_to_ledger(scan) -> str:
    Config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    ledger = read_ledger()
    previous_hash = ledger[-1]["current_hash"] if ledger else "GENESIS"
    current_hash = compute_scan_hash(scan, previous_hash)

    entry = {
        "entry_index": len(ledger) + 1,
        "scan_id": scan.id,
        "timestamp": datetime.utcnow().isoformat(),
        "previous_hash": previous_hash,
        "current_hash": current_hash
    }

    ledger.append(entry)

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    return current_hash