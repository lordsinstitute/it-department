from datetime import datetime
from detector import db, login_manager
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="USER", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

class AnalysisRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    input_type = db.Column(db.String(20), nullable=False)  # "text" or "file"
    filename = db.Column(db.String(255), nullable=True)

    snippet = db.Column(db.Text, nullable=True)  # short snippet for history
    findings_json = db.Column(db.Text, nullable=False)  # stored as json string

    risk_score = db.Column(db.Integer, nullable=False, default=0)
    risk_level = db.Column(db.String(20), nullable=False, default="Low")

    ledger_hash = db.Column(db.String(64), nullable=False, default="")
    ledger_prev_hash = db.Column(db.String(64), nullable=False, default="")