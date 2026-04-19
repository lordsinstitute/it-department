import os
import json
import uuid
from datetime import datetime

def now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": "Could not serialize safely."}, indent=2)

def secure_filename_basic(name: str) -> str:
    name = name.replace("\\", "_").replace("/", "_").strip()
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    clean = "".join(keep).strip().replace(" ", "_")
    if not clean:
        clean = "file"
    return clean[:120]

def secure_save_upload(file_storage, upload_folder: str) -> str:
    os.makedirs(upload_folder, exist_ok=True)
    original = secure_filename_basic(file_storage.filename or "upload")
    ext = os.path.splitext(original)[1].lower()
    token = uuid.uuid4().hex[:10]
    final_name = f"{os.path.splitext(original)[0]}_{token}{ext}"
    path = os.path.join(upload_folder, final_name)
    file_storage.save(path)
    return path

def safe_stat(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"size_bytes": int(st.st_size)}
    except Exception:
        return {"size_bytes": 0}