import json
import hashlib
from datetime import datetime
from pathlib import Path


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EvidenceLedger:
    """
    "Blockchain-inspired" JSON ledger:
    - Each entry has (index, timestamp, run_id, prev_hash, hash, digest)
    - hash = SHA256(index|timestamp|run_id|prev_hash|digest)
    - digest = SHA256(canonical_json(analysis_obj))
    Stored locally in ledger.json in app base dir.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        if not self.ledger_path.exists():
            self._write({"version": 1, "chain": []})

    def _read(self) -> dict:
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "chain": []}

    def _write(self, data: dict) -> None:
        self.ledger_path.write_text(_canonical_json(data), encoding="utf-8")

    @staticmethod
    def sha256_hex(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

    def append_run(self, *, run_id: int, analysis_obj: dict) -> dict:
        data = self._read()
        chain = data.get("chain", [])
        index = len(chain)
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        prev_hash = chain[-1]["hash"] if chain else "0" * 64

        digest = self.sha256_hex(_canonical_json(analysis_obj))
        material = f"{index}|{ts}|{run_id}|{prev_hash}|{digest}"
        h = self.sha256_hex(material)

        entry = {
            "index": index,
            "timestamp": ts,
            "run_id": run_id,
            "prev_hash": prev_hash,
            "digest": digest,
            "hash": h,
        }
        chain.append(entry)
        data["chain"] = chain
        self._write(data)
        return {"index": index, "hash": h, "prev_hash": prev_hash, "digest": digest}