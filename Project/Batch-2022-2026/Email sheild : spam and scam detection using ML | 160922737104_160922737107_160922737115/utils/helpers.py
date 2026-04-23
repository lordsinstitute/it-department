ALLOWED_EXTENSIONS = {"txt", "eml"}


def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_read_uploaded_file(file_storage) -> str:
    try:
        raw = file_storage.read()
        if not raw:
            return ""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    except Exception:
        return ""