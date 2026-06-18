import json, hashlib, os

LEDGER = "ledger.json"

def verify_ledger():
    if not os.path.exists(LEDGER):
        return False, "Ledger missing"

    with open(LEDGER, "r") as f:
        ledger = json.load(f)

    previous = "0"
    for index, entry in enumerate(ledger):
        recalculated = hashlib.sha256(
            (str(entry["data"]) + previous).encode()
        ).hexdigest()

        if recalculated != entry["hash"]:
            return False, f"Tampering detected at block {index}"

        previous = entry["hash"]

    return True, "Ledger verified – no tampering detected"