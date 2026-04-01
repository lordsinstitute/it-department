from flask import *
import joblib
import numpy as np
from datetime import datetime
import pickle
import pandas as pd


user_bp = Blueprint('user_bp', __name__)

# Load trained model and encoders
model = joblib.load("Accident_model.pkl")
encoders = joblib.load("label_encoder.pkl")
feature_encoders = {
        col: encoders[col]
        for col in encoders
        if col != "Accident_severity"
    }


@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict.html",  encoders=feature_encoders)
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

@user_bp.route('/predict1',  methods=['POST', 'GET'])
def predict1():
    return render_template("predict.html",  encoders=feature_encoders)


@user_bp.route('/predict',  methods=['POST', 'GET'])
def predict():
    prediction = None

    if request.method == "POST":
        input_data = {}

        for col, le in feature_encoders.items():
            value = request.form.get(col)
            input_data[col] = le.transform([value])[0]

        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        prediction = encoders["Accident_severity"].inverse_transform([pred])#[0]
        pred_class = np.argmax(proba)
        print(pred_class)
        if pred_class==0:
            prediction="Slight Injury"
        else:
            prediction="Severe Injury"

    return render_template(
        "predict.html",
        encoders=feature_encoders,
        prediction=prediction,
        prob=round(np.max(proba)*100,2)
    )


@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")