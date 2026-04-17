from flask import Blueprint, render_template, request
from views import preprocess

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/user')
def user():
    return render_template("user-login.html")

@user_bp.route('/user_home', methods=['POST', 'GET'])
def user_home():
    msg = ''
    if request.form.get('adminUser') == 'user' and request.form.get('adminPass') == 'user':
        return render_template("index.html", link=False, error=None)
    else:
        msg = 'Incorrect username or password!'
    return render_template('user-login.html', msg=msg)

@user_bp.route('/logout')
def logout():
    return render_template("home.html")