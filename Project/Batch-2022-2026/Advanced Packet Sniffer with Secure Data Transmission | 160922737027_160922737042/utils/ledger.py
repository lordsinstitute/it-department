import hashlib
import json
import os
from datetime import datetime


class Ledger:
    """
    Blockchain-inspired local evidence ledger:
    - Stores a JSON array of blocks
    - Each block has: index, timestamp, prev_hash, hash, payload_hash
    - The block hash is SHA256(index + timestamp + prev_hash + payload_hash)
    """

    def __init__(self, ledger_path: str, logger=None):
        self.ledger_path = ledger_path
        self.logger = logger
        self._ensure_ledger()

    def _ensure_ledger(self):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                f.write("[]")

    def _read(self):
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Repair corruption by resetting to empty
            if self.logger:
                self.logger.exception("Ledger read failed; resetting to empty ledger.")
            return []

    def _write(self, blocks):
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

    @staticmethod
    def sha256_hex(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def append(self, payload: dict) -> dict:
        blocks = self._read()

        index = len(blocks)
        timestamp = datetime.utcnow().isoformat() + "Z"

        prev_hash = blocks[-1]["hash"] if blocks else None

        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_hash = self.sha256_hex(payload_json)

        block_material = f"{index}|{timestamp}|{prev_hash or ''}|{payload_hash}"
        block_hash = self.sha256_hex(block_material)

        block = {
            "index": index,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
            "payload_hash": payload_hash,
            "hash": block_hash,
        }

        blocks.append(block)
        try:
            self._write(blocks)
        except Exception:
            if self.logger:
                self.logger.exception("Ledger write failed.")
            # Still return computed hashes to keep app running
        return {"index": index, "hash": block_hash, "prev_hash": prev_hash, "payload_hash": payload_hash}