import os, math

def entropy(data):
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    ent = 0
    for c in freq.values():
        p = c / len(data)
        ent -= p * math.log2(p)
    return ent

def scan_process_memory(pid):
    mem_path = f"/proc/{pid}/mem"

    if not os.path.exists(mem_path):
        return {"error": "Memory access not permitted"}

    try:
        with open(mem_path, "rb") as f:
            chunk = f.read(4096)
            ent = entropy(chunk)
    except:
        return {"error": "Access denied"}

    suspicious = ent > 7.5
    return {
        "pid": pid,
        "entropy": round(ent, 3),
        "suspicious": suspicious
    }