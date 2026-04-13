from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from config import Config
import os

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"

def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)

    # Ensure folders exist (uploads, database)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["DATABASE_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from detector.models import User, AnalysisRun  # noqa
    from detector.auth import auth_bp
    from detector.routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # DB auto-create + create default admin safely
    with app.app_context():
        db.create_all()
        _ensure_default_admin()

    # Never crash on demo: friendly error handlers
    @app.errorhandler(413)
    def too_large(_e):
        from flask import render_template
        return render_template("error.html", title="File Too Large",
                               message="Upload too large. Max allowed is 5 MB."), 413

    @app.errorhandler(404)
    def not_found(_e):
        from flask import render_template
        return render_template("error.html", title="Not Found",
                               message="That page does not exist."), 404

    @app.errorhandler(500)
    def server_error(_e):
        from flask import render_template
        return render_template("error.html", title="Server Error",
                               message="Something went wrong, but the app is still running. Try again."), 500

    return app

def _ensure_default_admin():
    from detector.models import User
    from werkzeug.security import generate_password_hash
    from detector import db

    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(
            username="admin",
            password_hash=generate_password_hash("admin123!", method="pbkdf2:sha256", salt_length=16),
            role="ADMIN"
        )
        db.session.add(admin)
        db.session.commit()
        print("[BOOTSTRAP] Created default ADMIN -> username: admin | password: admin123!")