from flask import Flask, render_template, request, redirect, url_for, session
import os
import joblib
import numpy as np
import os
app = Flask(__name__)
app.secret_key = "ddos_secret_key"

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------------
# ADMIN MODULE
# -------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    session.clear()
    session['admin'] = True
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["role"] = "admin"
            return redirect(url_for("admin_eda"))
    return render_template("admin_login.html")


@app.route("/admin/eda")
def admin_eda():
    eda_path = os.path.join(app.static_folder, "eda")
    images = [
        f for f in os.listdir(eda_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    images.sort()

    return render_template(
        "admin_eda.html",
        eda_images=images
    )

@app.route("/admin/model-comparison")
def admin_compare():
    import os
    import pandas as pd

    eval_path = os.path.join(app.static_folder, "evaluation")

    # Confusion matrices
    cm_path = os.path.join(eval_path, "confusion_matrices")
    cm_images = sorted(os.listdir(cm_path))

    # Classification reports
    report_path = os.path.join(eval_path, "classification_reports")
    reports = {}
    for file in sorted(os.listdir(report_path)):
        model_name = file.replace("_report.csv", "")
        df = pd.read_csv(os.path.join(report_path, file))
        reports[model_name] = df

    return render_template(
        "admin_compare.html",
        cm_images=cm_images,
        reports=reports
    )


@app.route("/admin/final-model")
def admin_final_model():
    import pandas as pd
    import os

    model_dir = os.path.join(app.root_path, "static/final_model")

    # Metrics
    metrics_df = pd.read_csv(os.path.join(model_dir, "metrics.csv"))
    metrics = metrics_df.iloc[0].to_dict()

    # Classification report (text)
    with open(os.path.join(model_dir, "classification_report.txt")) as f:
        report_text = f.read()

    return render_template(
        "admin_final_model.html",
        metrics=metrics,
        report_text=report_text
    )


# -------------------------
# USER MODULE
# -------------------------
@app.route("/user", methods=["GET", "POST"])
def user_login():
    session.clear()
    session['user'] = True
    if request.method == "POST":
        if request.form["username"] == "user" and request.form["password"] == "user":
            session["role"] = "user"
            return redirect(url_for("user_predict"))
    return render_template("user_login.html")

@app.route("/user/predict", methods=["GET", "POST"])
def user_predict():
    model_dir = os.path.join(app.root_path, "models")

    # Load artifacts (ONCE PER REQUEST, SAFE)
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_order = joblib.load(os.path.join(model_dir, "feature_order.pkl"))

    prediction = None
    probability = None
    confidence = None

    if request.method == "POST":
        input_values = []

        for feature in feature_order:
            value = request.form.get(feature)
            if value is None or value.strip() == "":
                return "Missing input for feature: " + feature, 400
            input_values.append(float(value))

        # Convert to correct shape (1, n_features)
        X_input = np.array(input_values).reshape(1, -1)

        # Apply SAME scaler (NO refit)
        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)[0]

        # 🔹 Probability (CONFIDENCE)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = round(np.max(probabilities) , 2)
        else:
            confidence = None

    return render_template(
        "user_predict.html",
        features=feature_order,
        prediction=prediction,
        probability=confidence
    )


@app.route("/user/result")
def user_result():
    return render_template("user_result.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
