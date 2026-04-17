from flask import *
import data.FinalClassifier, data.DataAnalysis
from os import path
from views import preprocess


admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/admin')
def admin():
    return render_template("admin.html")


@admin_bp.route('/admin_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'admin' and request.form['pwd'] == 'admin':
        return render_template("upload_dataset.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('admin.html', msg=msg)


@admin_bp.route('/admin_upload',  methods=['POST', 'GET'])
def admin_upload():
    msg = ''
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        msg = f.filename+' uploaded successfully'
        return render_template("upload_dataset.html", name=f.filename, msg=msg)
    else:
        return render_template("upload_dataset.html")


@admin_bp.route('/data_analysis')
def data_analysis():
    #data.DataAnalysis.dataAnalysis()
    return render_template("DataAnalysis.html")


@admin_bp.route('/eval_alg')
def eval_alg():
    #acc = data.CompareAlgorithms.compAlg()
    return render_template("CompAlg.html")


@admin_bp.route('/cr_model')
def cr_model():
    #msg, acc = data.FinalClassifer.createModel()
    msg = ""
    acc = 0
    msg_pp = preprocess()

    if (path.exists('models/CNN_LSTM_hybrid_model.h5') and msg_pp == "valid"):
        msg = "CNN LSTM Hybrid Model created successfully."
        return render_template("create_model.html", msg=msg)
    else:
        msg = ""
        return render_template("create_model.html", msg=msg)
    return render_template("create_model.html", msg=msg)




@admin_bp.route('/logout')
def logout():
    return render_template("home.html")