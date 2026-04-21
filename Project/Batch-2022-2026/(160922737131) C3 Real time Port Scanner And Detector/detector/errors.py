from flask import render_template

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(e):
        return render_template("error.html", code=413, message="Uploaded file is too large."), 413

    @app.errorhandler(500)
    def server_error(e):
        # Don’t leak stack traces; keep demo stable
        return render_template("error.html", code=500, message="Something went wrong. Please try again."), 500