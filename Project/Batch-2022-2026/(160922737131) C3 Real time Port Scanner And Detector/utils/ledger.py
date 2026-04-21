import json
import os
import hashlib
from datetime import datetime

def ensure_ledger_exists(ledger_path: str):
    if not os.path.exists(ledger_path):
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "created_at": datetime.utcnow().isoformat(), "chain": []}, f, indent=2)

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def append_ledger_entry(ledger_path: str, run_id: int, created_at: str, target: str, findings_json: str) -> str:
    """
    Blockchain-inspired evidence integrity:
    - result_hash = SHA256(findings_json)
    - chain_hash = SHA256(prev_hash + result_hash + metadata)
    Stored in JSON ledger (append-only style).
    Returns chain_hash.
    """
    ensure_ledger_exists(ledger_path)

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    chain = ledger.get("chain", [])
    prev_hash = chain[-1]["chain_hash"] if chain else "0" * 64

    result_hash = _sha256_hex(findings_json.encode("utf-8", errors="ignore"))

    entry_core = {
        "run_id": run_id,
        "created_at": created_at,
        "target": target,
        "result_hash": result_hash,
        "prev_hash": prev_hash,
    }

    chain_hash = _sha256_hex((_canonical_json(entry_core)).encode("utf-8"))
    entry = {**entry_core, "chain_hash": chain_hash}

    chain.append(entry)
    ledger["chain"] = chain

    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    return chain_hash