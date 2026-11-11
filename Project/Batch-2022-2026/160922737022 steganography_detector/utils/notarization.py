import hashlib, json, time, os

NOTARY_FILE = "ledger_notary.json"

def notarize_ledger(ledger_path="ledger.json"):
    if not os.path.exists(ledger_path):
        return None

    with open(ledger_path, "rb") as f:
        ledger_data = f.read()

    timestamp = int(time.time())
    hash_value = hashlib.sha256(ledger_data).hexdigest()

    record = {
        "ledger_hash": hash_value,
        "timestamp": timestamp,
        "note": "Offline notarization anchor"
    }

    history = []
    if os.path.exists(NOTARY_FILE):
        history = json.load(open(NOTARY_FILE))

    history.append(record)

    with open(NOTARY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return record