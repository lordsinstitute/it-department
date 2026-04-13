import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("training/best_ecg_cnn_model.h5")

CLASS_NAMES = [
    "Myocardial Infarction",
    "History of MI",
    "Abnormal Heartbeat",
    "Normal ECG"
]

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cnn")
def cnn():
    # Example metrics (replace with your real values if stored)
    # Load classification report
    report_path = "static/evaluation/cnn/classification_report_cnn.txt"
    with open(report_path, "r") as f:
        classification_report_text = f.read()

    metrics_data = {
        "accuracy": "97%",
        "loss": "0.08",
        "f1_score": "0.96"
    }
    return render_template("cnn.html", metrics=metrics_data, classification_report=classification_report_text)


@app.route("/resnet")
def resnet():
    # Example metrics (replace with your real values if stored)
    # Load classification report
    report_path = "static/evaluation/resnet/classification_report_resnet.txt"
    with open(report_path, "r") as f:
        classification_report_text = f.read()

    metrics_data = {
        "accuracy": "35%",
        "loss": "1.3176",
        "f1_score": "0.24"
    }
    return render_template("resnet.html", metrics=metrics_data, classification_report=classification_report_text)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["ecg_image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            class_index = np.argmax(preds)
            prediction = CLASS_NAMES[class_index]
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence
    )

# -------------------- MAIN --------------------

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
