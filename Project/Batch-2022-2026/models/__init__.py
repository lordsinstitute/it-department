# models/__init__.py

from datetime import datetime
from database import db  # IMPORTANT: use the single shared db instance from database package


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    scans = db.relationship(
        'Scan',
        backref='user',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<User {self.username}>'


class Scan(db.Model):
    __tablename__ = 'scans'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500))
    risk_level = db.Column(db.String(20), nullable=False)
    threat_count = db.Column(db.Integer, default=0)
    analysis_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    results = db.relationship(
        'AnalysisResult',
        backref='scan',
        lazy='joined',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<Scan {self.id}: {self.filename}>'


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False, index=True)
    threats = db.Column(db.Text)
    pdf_info = db.Column(db.Text)  # keep your fix: pdf_info (not metadata)
    hash_value = db.Column(db.String(64), unique=True)
    hash_chain = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<AnalysisResult {self.id}>'