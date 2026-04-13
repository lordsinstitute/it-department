from flask import Blueprint

admin_bp = Blueprint(
    "admin",
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

from admin import routes  # noqa: E402
