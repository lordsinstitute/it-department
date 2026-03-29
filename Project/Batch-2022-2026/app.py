import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =============================
# CONFIGURATION
# =============================
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/final_inc_model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load InceptionV3 Model
model = load_model(MODEL_PATH)

# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/cnn")
def cnn_performance():
    cnn_folder = "static/cnn"
    files = os.listdir(cnn_folder)
    return render_template("cnn.html", files=files)


@app.route("/inception")
def inception_performance():
    inc_folder = "static/inc"
    files = os.listdir(inc_folder)
    return render_template("inception.html", files=files)


@app.route("/best_model")
def best_model():
    inc_folder = "static/inc"
    files = os.listdir(inc_folder)
    return render_template("best_model.html", files=files)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)[0][0]

            if pred > 0.5:
                prediction = "Hemorrhage Detected"
                confidence = round(float(pred) * 100, 2)
            else:
                prediction = "No Hemorrhage"
                confidence = round((1 - float(pred)) * 100, 2)

    return render_template("predict.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)