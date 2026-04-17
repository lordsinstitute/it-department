from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(password: str) -> str:
    # Werkzeug recommended default uses PBKDF2:sha256 (secure + no external deps)
    return generate_password_hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return check_password_hash(stored_hash, password)
    except Exception:
        return False