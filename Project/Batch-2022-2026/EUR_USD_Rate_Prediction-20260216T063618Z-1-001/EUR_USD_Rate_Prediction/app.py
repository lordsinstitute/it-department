from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')


# -------------------------------------------------
# HOME
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------------------------------------
# UPLOAD DATASET
# -------------------------------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    message = None
    status = None

    if request.method == 'POST':
        file = request.files.get('dataset')

        if file and file.filename.endswith('.csv'):
            upload_dir = os.path.join(app.root_path, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)

            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)

            message = f"Dataset '{file.filename}' uploaded successfully."
            status = "success"
        else:
            message = "Please upload a valid CSV file."
            status = "error"

    return render_template(
        'upload.html',
        message=message,
        status=status
    )

# -------------------------------------------------
# EDA PAGE
# -------------------------------------------------
@app.route('/eda')
def eda():
    images = os.listdir(os.path.join(STATIC_DIR, 'eda'))
    return render_template('eda.html', images=images)

# -------------------------------------------------
# PERFORMANCE PAGE
# -------------------------------------------------
@app.route('/performance')
def performance():
    metrics_path = os.path.join(app.static_folder, 'performance', 'model_performance_metrics.csv')
    df = pd.read_csv(metrics_path)

    return render_template(
        'performance.html',
        metrics=df.to_dict(orient='records')
    )

# -------------------------------------------------
# LSTM PERFORMANCE
# -------------------------------------------------
@app.route('/lstm')
def lstm():
    metrics_path = os.path.join(app.static_folder, 'lstm_v1', 'gru_metrics.csv')
    df = pd.read_csv(metrics_path)

    images = [
        'prediction_vs_actual.png',
        'train_actual_predicted_overlap.png',
        'training_loss.png'
    ]

    return render_template(
        'lstm.html',
        metrics=df.to_dict(orient='records'),
        images=images
    )


# -------------------------------------------------
# LINEAR REGRESSION PERFORMANCE
# -------------------------------------------------
@app.route('/linear-regression')
def linear_regression():
    metrics_path = os.path.join(
        app.static_folder, 'outputs_lr', 'linear_regression_metrics.csv'
    )
    df = pd.read_csv(metrics_path)

    images = [
        'linear_regression_prediction.png',
        'train_actual_predicted_overlap.png'
    ]

    return render_template(
        'linear_regression.html',
        metrics=df.to_dict(orient='records'),
        images=images
    )

# -------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        close = float(request.form['close'])
        sma = float(request.form['sma'])
        ema = float(request.form['ema'])


        model = joblib.load('static/outputs_lr/linear_regression_model.pkl')
        scaler = joblib.load('static/outputs_lr/scaler.pkl')


        X = scaler.transform([[close, sma, ema]])
        prediction = model.predict(X)[0]


    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)