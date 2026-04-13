from flask import *
from views import preprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/user')
def user():
    return render_template("user.html")

@user_bp.route('/user_home', methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['User'] == 'user' and request.form['Pass'] == 'user':
        return render_template("predict.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

@user_bp.route('/predict1', methods=['GET', 'POST'])
def predict1():
    return render_template("predict.html")

@user_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = "models/best_xcp_model.h5"
    model = tf.keras.models.load_model(model_path)
    IMAGE_SIZE = (128, 128)
    CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
    prediction_label = ""
    confidence=0
    if request.method == 'POST':
        if(preprocess()=="valid"):
            file = request.files['image']
            if file:
                #filepath = 'static/uploads/test.jpg'
                #print(filepath)
                file.save("static/uploads/test.jpg")

                file.stream.seek(0)  # Reset the stream position
                img = load_img(io.BytesIO(file.read()), target_size=IMAGE_SIZE)
                img_array = img_to_array(img) / 255.0  # Normalize as during training
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = np.max(prediction)
                print(file.filename)

                plt.imshow(img)
                plt.title(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")
                plt.axis('off')
                plt.savefig('static/predictions/result.jpg')

                return render_template('predict.html', filename=file.filename, pred=predicted_class, conf=round(confidence*100,2))
        else:
            return render_template('predict.html', filename=None)
    return render_template('predict.html', filename=None)

@user_bp.route('/logout')
def logout():
    return render_template("Home.html")

