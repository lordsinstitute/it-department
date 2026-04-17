from flask import *
#import data.FinalClassifier
from views import preprocess


admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/admin')
def admin():
    return render_template("admin-login.html")


@admin_bp.route('/admin_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    print("hi1")
    if(preprocess()=="valid"):
        print("hi2")
        if request.form['adminUser'] == 'admin' and request.form['adminPass'] == 'admin':
            return render_template("admin-home.html")
        else:
            msg = 'Incorrect username / password !'
    return render_template('admin-login.html', msg=msg)

@admin_bp.route('/acc')
def acc():
    return render_template("acc.html")


@admin_bp.route('/loss')
def loss():
    return render_template("loss.html")

@admin_bp.route('/cnf')
def cnf():
    return render_template("cnf.html")

@admin_bp.route('/clf_rpt')
def clf_rpt():
    return render_template("clf_rpt.html")

@admin_bp.route('/roc')
def roc():
    return render_template("roc.html")


@admin_bp.route('/prc')
def prc():
    return render_template("prc.html")

@admin_bp.route('/logout')
def logout():
    return render_template("Index.html")

