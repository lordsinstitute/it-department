from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ===============================
# Load Model
# ===============================
model = joblib.load("static/final_model/best_xgboost_model.pkl")

FEATURE_COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_meds", "sick", "pregnant", "thyroid_surgery",
    "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
    "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH", "T3", "TT4", "T4U", "FTI"
]

TARGET_LABELS = {
    0: "Negative",
    1: "Hypothyroid",
    2: "Hyperthyroid"
}

# ===============================
# Home → Predict Page
# ===============================
@app.route("/")
def home():
    return render_template("predict.html")

# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {}

        for col in FEATURE_COLUMNS:
            input_data[col] = float(request.form[col])

        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]

        prediction = model.predict(input_df)[0]
        result = TARGET_LABELS[prediction]

        return render_template(
            "predict.html",
            prediction=result
        )

    except Exception as e:
        return render_template(
            "predict.html",
            prediction=f"Error: {str(e)}"
        )

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
