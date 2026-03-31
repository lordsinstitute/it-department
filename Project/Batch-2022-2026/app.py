from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("best_resnet_model.h5")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prob = model.predict(img)[0][0]

        if prob > 0.5:
            prediction = "Abnormal (Malnutrition Detected)"
            confidence = round(prob * 100, 2)
        else:
            prediction = "Normal (Healthy Child)"
            confidence = round((1 - prob) * 100, 2)

    return render_template("predict.html", prediction=prediction, confidence=confidence)

@app.route("/metrics")
def metrics():
    return render_template("metrics.html")

if __name__ == "__main__":
    app.run(debug=True)
