from flask import *
import joblib
import numpy as np
from datetime import datetime
import joblib
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
        model = joblib.load('California_Model.pkl')
        float_ = [float(x) for x in request.form.values()]
        final_ = [np.array(float_)]
        prediction = model.predict(final_)
        output = round(prediction[0], 2)
        pred = 'Affording home in California is  $ {}'.format(output)

        return (render_template("result1.html", prediction_text=pred))


@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")