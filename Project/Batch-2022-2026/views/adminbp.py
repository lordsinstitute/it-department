from flask import *
from werkzeug.utils import secure_filename
import data.FinalClassifier
import data.DataAnalysis
import data.CompareAlgorithms
from views import preprocess
import os

admin_bp = Blueprint('admin_bp', __name__)

# Project root directory (one level up from views/)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@admin_bp.route('/admin')
def admin():
    return render_template("admin.html")


@admin_bp.route('/admin_home', methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'admin' and request.form['pwd'] == 'admin':
        return render_template("upload_dataset.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('admin.html', msg=msg)


@admin_bp.route('/admin_upload', methods=['POST', 'GET'])
def admin_upload():
    msg = ''
    if request.method == 'POST':
        f = request.files.get('file')

        # Check if a file was actually selected
        if not f or f.filename == '':
            msg = 'No file selected. Please choose a file to upload.'
            return render_template("upload_dataset.html", msg=msg)

        # Save file safely to the project root directory
        filename = secure_filename(f.filename)
        save_path = os.path.join(PROJECT_DIR, filename)
        f.save(save_path)

        msg = 'File uploaded successfully'
        return render_template("upload_dataset.html", name=filename, msg=msg)
    else:
        return render_template("upload_dataset.html")


@admin_bp.route('/data_analysis')
def data_analysis():
    data.DataAnalysis.dataAnalysis()
    return render_template("DataAnalysis.html")


@admin_bp.route('/eval_alg')
def eval_alg():
    acc = data.CompareAlgorithms.compAlg()
    return render_template("CompAlg.html", acc=acc)


@admin_bp.route('/cr_model')
def cr_model():
    msg = ""
    acc = 0
    msg_pp = preprocess()

    if msg_pp == "valid":
        msg, acc = data.FinalClassifier.createModel()
        return render_template("create_model.html", msg=msg, acc=acc)
    else:
        msg = "Some error occurred"
        return render_template("create_model.html", msg=msg, acc=acc)


@admin_bp.route('/logout')
def logout():
    return render_template("home.html")