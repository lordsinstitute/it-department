from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    runs = db.relationship("AnalysisRun", backref="user", lazy=True)


class AnalysisRun(db.Model):
    __tablename__ = "analysis_runs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    source = db.Column(db.String(255), nullable=False)  # filename or "text-input"
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    risk_score = db.Column(db.Float, default=0.0, nullable=False)
    risk_level = db.Column(db.String(20), default="Low", nullable=False)

    result_json = db.Column(db.Text, nullable=False)

    # Evidence integrity / hash chain
    evidence_hash = db.Column(db.String(64), nullable=True)
    prev_hash = db.Column(db.String(64), nullable=True)
    chain_index = db.Column(db.Integer, nullable=True)