from flask import *
import os

admin_bp = Blueprint('admin_bp', __name__)


@admin_bp.route('/admin')
def admin():
    return render_template("admin.html")

@admin_bp.route('/admin_home', methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['adminUser'] == 'admin' and request.form['adminPass'] == 'admin':
        return render_template("mnet_acc.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('admin.html', msg=msg)

@admin_bp.route('/mnet_acc')
def mnet_acc():
    return render_template("mnet_acc.html")

@admin_bp.route('/mnet_loss')
def mnet_loss():
    return render_template("mnet_loss.html")

@admin_bp.route('/mnet_cnf')
def mnet_cnf():
    return render_template("mnet_cnf.html")

@admin_bp.route('/mnet_clfrpt')
def mnet_clfrpt():
    return render_template("mnet_clfrpt.html")

@admin_bp.route('/mnet_roc')
def mnet_roc():
    return render_template("mnet_roc.html")

@admin_bp.route('/mnet_prc')
def mnet_prc():
    return render_template("mnet_prc.html")

#################################################

@admin_bp.route('/incv3_acc')
def incv3_acc():
    return render_template("incv3_acc.html")

@admin_bp.route('/incv3_loss')
def incv3_loss():
    return render_template("incv3_loss.html")

@admin_bp.route('/incv3_cnf')
def incv3_cnf():
    return render_template("incv3_cnf.html")

@admin_bp.route('/incv3_clfrpt')
def incv3_clfrpt():
    return render_template("incv3_clfrpt.html")

@admin_bp.route('/incv3_roc')
def incv3_roc():
    return render_template("incv3_roc.html")

@admin_bp.route('/incv3_prc')
def incv3_prc():
    return render_template("incv3_prc.html")
###################################################

@admin_bp.route('/xcp_acc')
def xcp_acc():
    return render_template("xcp_acc.html")

@admin_bp.route('/xcp_loss')
def xcp_loss():
    return render_template("xcp_loss.html")

@admin_bp.route('/xcp_cnf')
def xcp_cnf():
    return render_template("xcp_cnf.html")

@admin_bp.route('/xcp_clfrpt')
def xcp_clfrpt():
    return render_template("xcp_clfrpt.html")

@admin_bp.route('/xcp_roc')
def xcp_roc():
    return render_template("xcp_roc.html")

@admin_bp.route('/xcp_prc')
def xcp_prc():
    return render_template("xcp_prc.html")
##################################################

