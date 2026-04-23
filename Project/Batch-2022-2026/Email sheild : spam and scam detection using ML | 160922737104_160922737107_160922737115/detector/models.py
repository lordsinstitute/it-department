from datetime import datetime
from . import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False, default="User")
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    scans = db.relationship("ScanResult", backref="user", lazy=True)


class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    source_type = db.Column(db.String(20), nullable=False)
    original_filename = db.Column(db.String(255), nullable=True)

    input_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text, nullable=False)

    ml_label = db.Column(db.String(20), nullable=False, default="safe")
    prediction_label = db.Column(db.String(20), nullable=False, default="Safe")
    confidence_score = db.Column(db.Float, nullable=False, default=0.0)

    risk_score = db.Column(db.Integer, nullable=False, default=0)
    risk_level = db.Column(db.String(20), nullable=False, default="Low")

    suspicious_links = db.Column(db.Integer, nullable=False, default=0)
    urgent_words = db.Column(db.Integer, nullable=False, default=0)
    financial_words = db.Column(db.Integer, nullable=False, default=0)
    attachment_words = db.Column(db.Integer, nullable=False, default=0)
    impersonation_words = db.Column(db.Integer, nullable=False, default=0)

    summary = db.Column(db.Text, nullable=False, default="")
    finding_details = db.Column(db.Text, nullable=False, default="")

    evidence_hash = db.Column(db.String(64), nullable=False, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)