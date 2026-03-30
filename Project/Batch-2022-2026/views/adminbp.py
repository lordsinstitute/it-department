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
            print("hi3")
            return render_template("admin-home.html")
        else:
            msg = 'Incorrect username / password !'
    return render_template('admin-login.html', msg=msg)

@admin_bp.route('/mnet')
def mnet():
    return render_template("admin-home.html")

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


@admin_bp.route('/cnn_clfrpt')
def cnn_clfrpt():
    return render_template("cnn_clfrpt.html")

@admin_bp.route('/cnn_acc')
def cnn_acc():
    return render_template("cnn_acc.html")

@admin_bp.route('/cnn_cnf')
def cnn_cnf():
    return render_template("cnn_cnf.html")

@admin_bp.route('/cnn_roc')
def cnn_roc():
    return render_template("cnn_roc.html")


@admin_bp.route('/inv3_acc')
def inv3_acc():
    return render_template("inv3_acc.html")

@admin_bp.route('/inv3_cnf')
def inv3_cnf():
    return render_template("inv3_cnf.html")

@admin_bp.route('/inv3_roc')
def inv3_roc():
    return render_template("inv3_roc.html")

@admin_bp.route('/inv3_clfrpt')
def inv3_clfrpt():
    return render_template("inv3_clfrpt.html")

@admin_bp.route('/logout')
def logout():
    return render_template("Index.html")

