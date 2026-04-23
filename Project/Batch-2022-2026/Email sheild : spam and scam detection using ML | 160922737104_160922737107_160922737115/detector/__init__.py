from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

db = SQLAlchemy()


def create_app():
    from .config import Config, ensure_directories
    from .models import User

    ensure_directories()

    app = Flask(
        __name__,
        template_folder=str(Config.TEMPLATES_DIR),
        static_folder=str(Config.STATIC_DIR),
    )
    app.config.from_object(Config)

    db.init_app(app)

    from .auth import auth_bp
    from .routes import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    @app.errorhandler(404)
    def not_found(_error):
        try:
            return render_template(
                "error.html",
                title="Page Not Found",
                message="The requested page was not found."
            ), 404
        except Exception:
            return "<h2>404 - Page Not Found</h2>", 404

    @app.errorhandler(500)
    def internal_error(_error):
        db.session.rollback()
        try:
            return render_template(
                "error.html",
                title="Server Error",
                message="An internal error occurred. Please try again."
            ), 500
        except Exception:
            return "<h2>500 - Internal Server Error</h2><p>Template error.html not found.</p>", 500

    with app.app_context():
        db.create_all()

        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                full_name="System Administrator",
                password_hash=generate_password_hash("admin123!")
            )
            db.session.add(admin)
            db.session.commit()
            print("[BOOTSTRAP] Created default ADMIN -> username: admin | password: admin123!")

    return app