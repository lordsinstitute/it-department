from flask import *
import joblib
import numpy as np
from datetime import datetime
import pickle
from views import preprocess

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict1.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

@user_bp.route('/predict', methods=['POST', 'GET'])
def predict():
    if (preprocess() == "valid"):
        features = [int(x) for x in request.form.values()]
        print(features)

        Pkl_Filename = "rf_tuned.pkl"
        with open(Pkl_Filename, 'rb') as file:
            model = pickle.load(file)


        final = np.array(features).reshape((1, 6))
        print(final)
        pred = model.predict(final)[0]
        print(pred)

        if pred < 0:
            return render_template('result1.html', prediction='Error calculating Amount!')
        else:
            return render_template('result1.html', prediction='The Health Insurance Premium predicted is: Rs.{0:.3f}'.format(pred))
    else:
        return render_template('result1.html', prediction='Error calculating Amount!')

@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")