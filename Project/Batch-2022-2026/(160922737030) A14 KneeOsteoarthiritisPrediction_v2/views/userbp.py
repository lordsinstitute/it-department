from flask import *
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from views import preprocess
import data.FinalClassifier

user_bp = Blueprint('user_bp', __name__)



@user_bp.route('/user')
def user():
    return render_template("user-login.html")

@user_bp.route('/user_home',  methods=['POST', 'GET'])
def user_home():
    msg = ''
    if request.form['adminUser'] == 'user' and request.form['adminPass'] == 'user':
        return render_template("predict1.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user-login.html', msg=msg)



@user_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_label = ""
    confidence=0
    if request.method == 'POST':
        if(preprocess()=="valid"):
            file = request.files['image']
            if file:
                #filepath = 'static/uploads/test.jpg'
                #print(filepath)
                file.save("static/uploads/test.jpg")
                prediction_label, confidence = data.FinalClassifier.TestModel()
                print(file.filename)
                return render_template('predict1.html', filename=file.filename, conf=round(confidence*100,2), pred=prediction_label)
        else:
            return render_template('predict1.html', filename=None)
    return render_template('predict1.html', filename=None)

@user_bp.route('/logout')
def logout():
    return render_template("Home.html")

