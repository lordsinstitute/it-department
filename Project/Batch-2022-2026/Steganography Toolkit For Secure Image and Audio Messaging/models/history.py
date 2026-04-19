from models.db import db

class RunHistory(db.Model):
    __tablename__ = "run_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    run_type = db.Column(db.String(16), nullable=False, default="encode")  # always encode
    media_type = db.Column(db.String(16), nullable=False)                 # image / audio

    input_text_len = db.Column(db.Integer, nullable=False, default=0)

    input_filename = db.Column(db.String(255), nullable=False)
    output_filename = db.Column(db.String(255), nullable=True)

    risk_score = db.Column(db.Integer, nullable=False, default=10)
    risk_level = db.Column(db.String(16), nullable=False, default="Low")

    findings_json = db.Column(db.Text, nullable=True)

    # Evidence integrity chain
    ledger_hash = db.Column(db.String(64), nullable=False)
    ledger_prev_hash = db.Column(db.String(64), nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, index=True)