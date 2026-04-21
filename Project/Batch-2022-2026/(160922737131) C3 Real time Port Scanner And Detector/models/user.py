from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from detector.extensions import db
from detector.extensions import login_manager

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(40), default="User")

    @staticmethod
    def create_user(username: str, password: str, role: str = "User"):
        u = User(username=username, password_hash=generate_password_hash(password), role=role)
        return u

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None