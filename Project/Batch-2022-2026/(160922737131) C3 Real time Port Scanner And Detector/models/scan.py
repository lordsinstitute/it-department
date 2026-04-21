from datetime import datetime
from detector.extensions import db

class ScanRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    target = db.Column(db.String(255), nullable=False)
    scan_type = db.Column(db.String(80), default="TCP Connect")

    findings_json = db.Column(db.Text, nullable=False)

    risk_level = db.Column(db.String(20), default="Low")
    risk_score = db.Column(db.Integer, default=0)

    ledger_hash = db.Column(db.String(64), nullable=True)  # SHA256 chain hash

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    created_by = db.Column(db.String(80), default="unknown")