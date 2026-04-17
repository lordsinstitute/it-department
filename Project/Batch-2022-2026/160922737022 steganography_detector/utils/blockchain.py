import json, hashlib, os

LEDGER = "ledger.json"

def record_hash(data):
    previous = "0"
    if os.path.exists(LEDGER):
        ledger = json.load(open(LEDGER))
        previous = ledger[-1]["hash"]

    current_hash = hashlib.sha256((str(data) + previous).encode()).hexdigest()
    entry = {"data": data, "prev": previous, "hash": current_hash}

    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    ledger.append(entry)

    with open(LEDGER, "w") as f:
        json.dump(ledger, f, indent=2)

    return current_hash