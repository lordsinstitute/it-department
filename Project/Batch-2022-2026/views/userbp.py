from flask import *
import joblib
import numpy as np
from datetime import datetime
import data.FinalClassifier
from views import preprocess

user_bp = Blueprint('user_bp', __name__)


@user_bp.route('/user')
def user():
    return render_template("user.html")

@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

@user_bp.route('/predict1')
def predict1():
    return render_template("predict.html")


def getParameters():
    parameters = []
    print("param")
    parameters.append(float(request.form['lbe']))
    parameters.append(float(request.form['lb']))
    parameters.append(float(request.form['ac']))
    parameters.append(float(request.form['fm']))
    parameters.append(float(request.form['uc']))

    parameters.append(float(request.form['dl']))
    parameters.append(float(request.form['ds']))
    print(parameters)
    parameters.append(float(request.form['dp']))
    parameters.append(float(request.form['dr']))
    print(parameters)

    return parameters

@user_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Hi")
    if request.method == 'POST' and preprocess() == "valid":
        lst=getParameters()
        pred=data.FinalClassifier.testModel(lst)
        return render_template("predict.html",prediction_text=pred)
    else:
        pred="Some error occurred"
        return render_template("predict.html", prediction_text=pred)
    return render_template("predict.html")




@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")