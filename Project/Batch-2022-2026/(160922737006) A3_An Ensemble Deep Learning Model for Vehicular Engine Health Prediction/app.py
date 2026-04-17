from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib, os

app = Flask(__name__)
app.secret_key = "engine_health_secret"


model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp"
]

@app.route("/")
def home():
    return render_template("home.html")

# ---------------- ADMIN ----------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    return render_template("admin_dashboard.html")

import os

@app.route("/admin/eda")
def eda():
    eda_path = os.path.join(app.root_path, "static", "EDA")
    images = []

    if os.path.exists(eda_path):
        images = [
            img for img in os.listdir(eda_path)
            if img.endswith(".png")
        ]

    return render_template("eda.html", images=images)


@app.route("/admin/performance")
def performance():
    perf_dir = os.path.join(app.root_path, "static", "Performance")

    reports = {}
    for file in os.listdir(perf_dir):
        if file.endswith("_report.txt"):
            model_name = file.replace("_report.txt", "")
            with open(os.path.join(perf_dir, file)) as f:
                reports[model_name] = f.read()

    return render_template("performance.html", reports=reports)

@app.route('/admin/dl_performance')
def dl_performance():

    folder_path = os.path.join('static', 'dl_model_outputs')

    # Collect image files
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Read text files
    text_files = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r') as f:
                text_files[file] = f.read()

    return render_template(
        'dl_performance.html',
        images=images,
        text_files=text_files
    )

import os
import pandas as pd
from flask import render_template

@app.route('/admin/stacked_performance')
def stacked_performance():

    folder_path = os.path.join('static', 'stacked_outputs')

    reports = {}
    confusion_matrices = []
    comparison_graph = None

    for file in os.listdir(folder_path):

        # Classification Reports
        if file.endswith('_classification_report.txt'):
            model_name = file.replace('_classification_report.txt', '')
            with open(os.path.join(folder_path, file), 'r') as f:
                reports[model_name] = f.read()

        # Confusion Matrices
        elif file.endswith('_confusion_matrix.png'):
            confusion_matrices.append(file)

        # Comparison graph (if exists as png)
        elif 'all_models_performance_metrics' in file and file.endswith('.png'):
            comparison_graph = file

    return render_template(
        'stacked_performance.html',
        reports=reports,
        confusion_matrices=confusion_matrices,
        comparison_graph=comparison_graph
    )



@app.route("/admin/best_model")
def best_model():

    folder_path = os.path.join('static', 'Performance')

    # Read Random Forest classification report
    report_path = os.path.join(folder_path, 'Random Forest_report.txt')
    rf_report = ""

    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            rf_report = f.read()

    # Confusion matrix filename
    rf_confusion_matrix = 'Random Forest_cm.png'

    return render_template(
        "best_model.html",
        rf_report=rf_report,
        rf_confusion_matrix=rf_confusion_matrix
    )

# ---------------- USER ----------------
@app.route("/user", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        if request.form["username"] == "user" and request.form["password"] == "user":
            session["user"] = True
            return redirect(url_for("predict"))
    return render_template("user_login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        values = [float(request.form[f]) for f in FEATURES]
        scaled = scaler.transform([values])
        result = model.predict(scaled)[0]
        status = "Healthy Engine" if result == 1 else "Faulty Engine"
        return render_template("result.html", result=status)
    return render_template("predict.html", features=FEATURES)

if __name__ == "__main__":
    app.run(debug=True)
