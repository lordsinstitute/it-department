import os
import json
import pandas as pd
from flask import Flask, render_template, request
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load HuggingFace Model
# -----------------------------
classifier = pipeline("image-classification", model="./food_model")

# -----------------------------
# Load Nutrition Data
# -----------------------------
nutrition_df = pd.read_csv("nutrition.csv")
nutrition_df["label"] = nutrition_df["label"].str.lower()

# -----------------------------
# Helper Functions
# -----------------------------
def get_nutrition(food):
    food = food.lower().replace(" ", "_")
    match = nutrition_df[nutrition_df["label"] == food]
    if match.empty:
        return None
    return match.iloc[0].to_dict()

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/mobilenet")
def mobilenet():
    per_folder = os.path.join("static", "performance")

    files = {
        "Accuracy Curve": "accuracy_curve.png",
        "Loss Curve": "loss_curve.png",
        "ROC Curve": "roc_curve.png",
        "Classification Report": "classification_report.csv",
        "Confusion Matrix": "confusion_matrix.png"
    }

    return render_template(
        "mobilenet.html",
        files=files,
        selected_file=None
    )


@app.route("/mobilenet/view/<filename>")
def view_file(filename):
    files = {
        "Accuracy Curve": "accuracy_curve.png",
        "Loss Curve": "loss_curve.png",
        "ROC Curve": "roc_curve.png",
        "Classification Report": "classification_report.csv",
        "Confusion Matrix": "confusion_matrix.png"
    }

    file_path = os.path.join("static", "performance", filename)

    file_content = None
    table_data = None
    is_image = False

    if filename.endswith((".png", ".jpg", ".jpeg")):
        is_image = True

    elif filename.endswith(".csv"):
        df = pd.read_csv(file_path)
        table_data = df.to_html(classes="table table-striped table-bordered", index=False)

    elif filename.endswith(".txt"):
        with open(file_path, "r") as f:
            file_content = f.read()

    return render_template(
        "mobilenet.html",
        files=files,
        selected_file=filename,
        is_image=is_image,
        file_content=file_content,
        table_data=table_data
    )


@app.route("/hf_performance")
def hf_performance():
    with open("food_model/all_results.json") as f:
        results = json.load(f)
    return render_template("hf_performance.html", results=results)


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            result = classifier(path)
            food = result[0]["label"]
            confidence = round(result[0]["score"] * 100, 2)

            nutrition = get_nutrition(food)

            return render_template(
                "prediction.html",
                prediction=food.replace("_", " ").title(),
                confidence=confidence,
                nutrition=nutrition,
                image=filename,
            )

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)