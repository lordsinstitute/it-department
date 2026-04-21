import os
from flask import Flask
from config import Config
from detector.extensions import db, login_manager
from detector.errors import register_error_handlers
from detector.routes import main_bp
from detector.auth import auth_bp
from models.user import User

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)

    # Ensure folders exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # Errors
    register_error_handlers(app)

    # Create DB + default user
    with app.app_context():
        db.create_all()
        _ensure_default_admin()

    return app

def _ensure_default_admin():
    # Default admin to avoid setup friction (change in Settings page)
    # username: admin  password: admin123
    if not User.query.first():
        u = User.create_user(username="admin", password="admin123", role="Admin")
        db.session.add(u)
        db.session.commit()