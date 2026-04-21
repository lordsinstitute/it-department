import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, flash
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.secret_key = "bitcoin_secret"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "models/lstm/lstm_model.h5"

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            flash("File uploaded successfully!", "success")
            return redirect("/upload")
    return render_template("upload.html")


# ---------------- EDA ----------------
@app.route("/eda")
def eda():
    images = os.listdir("static/eda")
    return render_template("eda.html", images=images)


# ---------------- LSTM PERFORMANCE ----------------
@app.route("/lstm")
def lstm():
    images = os.listdir("static/lstm")

    metrics_path = "static/lstm/metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return render_template("lstm.html", images=images, metrics=metrics)


# ---------------- PREDICT ----------------

LOOKBACK = 30
FUTURE_DAYS = 60

@app.route("/predict", methods=["POST", "GET"])
def predict():
    graph_filename = None
    error = None

    if request.method == "POST":
        start_date = request.form["start_date"]

        df = pd.read_csv("bitcoin_last_10_years.csv")
        df["Date"] = pd.to_datetime(df["Date"])

        start_date = pd.to_datetime(start_date)

        if start_date not in df["Date"].values:
            return render_template("predict.html", error="Date not found")

        idx = df.index[df["Date"] == start_date][0]

        if idx < LOOKBACK or idx + FUTURE_DAYS >= len(df):
            return render_template("predict.html",
                                   error="Not enough data for 2 months comparison.")

        prices = df["Price"].values.reshape(-1,1)

        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]



        scaler = joblib.load("models/scaler.save")
        scaler.fit(train_data)

        scaled = scaler.transform(prices)

        # Get input sequence
        input_seq = scaled[idx-LOOKBACK:idx]



        model = load_model("models/lstm_60day_model.h5", compile=False)

        pred_scaled = model.predict(input_seq.reshape(1,LOOKBACK,1))
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

        actual = df["Price"].iloc[idx:idx+FUTURE_DAYS].values

        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(actual, label="Actual")
        plt.plot(pred, label="Predicted")
        plt.title("60-Day Bitcoin Forecast vs Actual")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()

        graph_filename = "60day_prediction.png"
        plt.savefig("static/lstm/" + graph_filename)
        plt.close()

    return render_template("predict.html",
                           graph_filename=graph_filename,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)