import json
import hashlib
from pathlib import Path
from datetime import datetime
from utils.helpers import json_dumps_safe

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def init_ledger_if_missing(ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        return
    ledger = {"schema": "stegtk-ledger-v2", "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z", "chain": []}
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

def _read_ledger(ledger_path: Path) -> dict:
    try:
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": "stegtk-ledger-v2", "created_at": "", "chain": []}

def _write_ledger(ledger_path: Path, ledger: dict) -> None:
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

def add_to_ledger(base_dir, evidence_payload: dict):
    """
    evidence_payload should avoid circular dependencies with output bytes.
    We store:
      - watermark_id (embedded in payload header)
      - options
      - input filename
      - output filename
      - input checksum (safe; doesn't depend on embedding output changes)
    Output checksum can still be stored in DB findings, but not required for ledger hash-chain.
    """
    ledger_path = Path(base_dir) / "database" / "ledger.json"
    init_ledger_if_missing(ledger_path)
    ledger = _read_ledger(ledger_path)

    chain = ledger.get("chain", [])
    prev_hash = chain[-1]["chain_hash"] if chain else None

    payload_json = json_dumps_safe(evidence_payload).encode("utf-8")
    record_hash = _sha256_hex(payload_json)

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    link = (prev_hash or "").encode("utf-8") + record_hash.encode("utf-8") + ts.encode("utf-8")
    chain_hash = _sha256_hex(link)

    entry = {
        "ts": ts,
        "prev_hash": prev_hash,
        "record_hash": record_hash,
        "chain_hash": chain_hash,
        "watermark_id": evidence_payload.get("watermark_id"),
        "summary": {
            "mode": "encode",
            "media_type": evidence_payload.get("media_type"),
            "input_file": evidence_payload.get("input_file"),
            "output_file": evidence_payload.get("output_file"),
            "redundancy_r": evidence_payload.get("redundancy_r"),
            "adaptive_embedding": evidence_payload.get("adaptive_embedding"),
        }
    }
    chain.append(entry)
    ledger["chain"] = chain
    _write_ledger(ledger_path, ledger)
    return chain_hash, prev_hash

def read_ledger_tail(base_dir, n: int = 5):
    ledger_path = Path(base_dir) / "database" / "ledger.json"
    init_ledger_if_missing(ledger_path)
    ledger = _read_ledger(ledger_path)
    return ledger.get("chain", [])[-n:]