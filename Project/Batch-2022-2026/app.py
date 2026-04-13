from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "mental_health_secret"

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "static", "final_model")

model = joblib.load(os.path.join(MODEL_PATH, "best_model.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_PATH, "label_encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_PATH, "target_encoder.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_PATH,"feature_columns.pkl"))

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- ADMIN ----------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            return redirect(url_for("admin_eda"))
    return render_template("admin_login.html")

@app.route("/admin/eda")
def admin_eda():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    eda_path = os.path.join("static", "eda")
    eda_images = sorted(os.listdir(eda_path))
    return render_template(
        "admin_eda.html",
        eda_images=eda_images
    )

@app.route("/admin/compare")
def admin_compare():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    scores_df = pd.read_csv("static/evaluation/model_scores.csv")

    return render_template(
        "admin_compare.html",
        tables=scores_df.to_dict(orient="records")
    )

@app.route("/admin/final")
def admin_final():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    # Load scores
    scores_df = pd.read_csv("static/final_model/model_scores.csv")
    scores = scores_df.iloc[0].to_dict()

    # Load classification report
    with open("static/final_model/classification_report.txt") as f:
        report = f.read()

    return render_template(
        "admin_final_model.html",
        scores=scores,
        report=report
    )

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("home"))

# ---------------- USER ----------------
@app.route("/user/login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        if request.form["username"] == "user" and request.form["password"] == "user":
            session["user"] = True
            return redirect(url_for("predict"))
    return render_template("user_login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("user_login"))

    if request.method == "POST":
        input_data = []
        # Collect inputs from form
        feature_order = [
            "Gender",
            "Country",
            "Occupation",
            "self_employed",
            "family_history",
            "treatment",
            "Days_Indoors",
            "Growing_Stress",
            "Changes_Habits",
            "Mental_Health_History",
            "Coping_Struggles",
            "Work_Interest",
            "Social_Weakness",
            "mental_health_interview",
            "care_options"
        ]

        # Collect form input
        input_data = {}
        for feature in feature_order:
            input_data[feature] = request.form.get(feature)

        input_df = pd.DataFrame([input_data])

        # 🔥 APPLY LABEL ENCODING (CRITICAL FIX)
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col].astype(str))

        # Predict
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

        return render_template(
            "result.html",
            prediction=prediction_label
        )
    else:
        return render_template("predict.html")

@app.route("/user/logout")
def user_logout():
    session.pop("user", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
