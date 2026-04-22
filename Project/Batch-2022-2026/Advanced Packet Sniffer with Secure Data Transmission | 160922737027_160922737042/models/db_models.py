from datetime import datetime
from flask_login import UserMixin

from models import db


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    runs = db.relationship("Run", backref="user", lazy=True)


class Run(db.Model):
    __tablename__ = "runs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    input_mode = db.Column(db.String(10), nullable=False)  # text/file
    source_file = db.Column(db.String(255), nullable=True)
    stored_file = db.Column(db.String(500), nullable=True)

    score = db.Column(db.Integer, nullable=False)
    risk = db.Column(db.String(20), nullable=False)

    summary_json = db.Column(db.Text, nullable=True)
    findings_json = db.Column(db.Text, nullable=True)
    indicators_json = db.Column(db.Text, nullable=True)

    # Evidence integrity (hash chain)
    evidence_hash = db.Column(db.String(64), nullable=False)
    prev_hash = db.Column(db.String(64), nullable=True)
    chain_index = db.Column(db.Integer, nullable=False, default=0)