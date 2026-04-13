import os
from flask import Flask, render_template, redirect, url_for, session
from database import init_db
from admin import admin_bp
from user import user_bp
from validate import preprocess

def create_app():
    app = Flask(__name__)
    app.secret_key = "supersecretkey"  # change in production

    # Create required folders if missing
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    # Initialize DB (creates attendance.db if missing)
    init_db()

    # Register Blueprints
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(user_bp, url_prefix="/user")

    # Home route
    @app.route("/")
    def home():
        if preprocess()=="valid":
            return render_template("home.html")
        else:
            return render_template("base.html")

    # Logout
    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
