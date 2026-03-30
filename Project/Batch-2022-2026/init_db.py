from database import init_db, add_admin
from utils import hash_password

if __name__ == "__main__":
    print("[INFO] Initializing database and creating admin account...")
    init_db()
    password_hash = hash_password("admin123")
    add_admin("admin", password_hash)
    print("[INFO] Database initialized successfully. Default login -> username: admin, password: admin123")
