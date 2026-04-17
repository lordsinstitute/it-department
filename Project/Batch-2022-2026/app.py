from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import subprocess
from data.DataAnalysis import dataAnalysis
from data.CompareAlgorithms import compareAlgorithms
from data.FinalClassifier import createModel

app = Flask(__name__)
app.secret_key = "parkinsons_secret"

MODEL_PATH = "parkinson_classifier_model.pkl"
SCALER_PATH = "parkinson_scaler.pkl"

# ===============================
# Home
# ===============================
@app.route("/")
def home():
    return render_template("home.html")

# ===============================
# Admin Module
# ===============================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    session.clear()
    session['admin'] = True
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html")

@app.route("/admin/eda")
def admin_eda():
    dataAnalysis()
    return render_template("eda.html")

@app.route("/admin/compare")
def admin_compare():
    compareAlgorithms()
    return render_template("compare.html")

@app.route("/admin/final")
def admin_final():
    createModel()
    return render_template("final_model.html")

# ===============================
# User Module
# ===============================
@app.route("/user/login", methods=["GET", "POST"])
def user_login():
    session.clear()
    session['user'] = True
    if request.method == "POST":
        if request.form["username"] == "user" and request.form["password"] == "user":
            session["user"] = True
            return redirect(url_for("predict"))
    return render_template("user_login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    if "user" not in session:
        return redirect(url_for("user_login"))

    if request.method == "POST":
        features_order = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
            "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
            "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]

        input_data = np.array([float(request.form[f]) for f in features_order]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "Parkinson’s Detected" if prediction == 1 else "Healthy"
        return render_template("result.html", result=result)

    return render_template("predict.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
