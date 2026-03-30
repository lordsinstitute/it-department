from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(password):
    """Generate hashed password"""
    return generate_password_hash(password)


def verify_password(password, hashed):
    """Check password against hash"""
    return check_password_hash(hashed, password)
