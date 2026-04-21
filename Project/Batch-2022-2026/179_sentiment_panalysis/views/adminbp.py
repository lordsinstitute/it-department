from flask import Blueprint, render_template, request
from views import preprocess

admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/admin')
def admin():
    return render_template("admin-login.html")

@admin_bp.route('/admin_home', methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if preprocess() == "valid":
        if request.form.get('adminUser') == 'admin' and request.form.get('adminPass') == 'admin':
            return render_template("model_performance.html")
        else:
            msg = 'Incorrect username or password!'
    return render_template('admin-login.html', msg=msg)

@admin_bp.route('/logout')
def logout():
    return render_template("home.html")