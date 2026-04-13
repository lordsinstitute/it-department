from utils.custom_model import CustomModelling, load_checkpoint
from pathlib import Path
from flask import *
import views.adminbp, views.userbp
import pandas as pd
from utils.dataset import feature_engineering
app = Flask(__name__)
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

def load_model():
    model_path = Path("model_best.pkl")
    print(model_path)
    model = load_checkpoint(model_path)
    return model


custom_model = CustomModelling(
    model=load_model(), x_train=None, y_train=None, x_test=None
)

class_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II",
}





def predict_sample(sample: dict) -> dict:
    sample =sample['data']
    sample = [sample]
    sample_df = pd.DataFrame(sample)

    feature = feature_engineering(data=sample_df)

    predictions = custom_model.inference(feature)
    return {"prediction": predictions.item(), "class": class_names[predictions.item()]}


def create_request_data(request: request) -> dict:  # type: ignore
    gender = request.form.get("gender")
    age = int(request.form.get("age"))
    height = float(request.form.get("height").replace(",", ".")) / 100
    weight = float(request.form.get("weight"))
    fam = request.form.get("fam")
    favc = request.form.get("favc")
    fcvc = request.form.get("fcvc")
    ncp = request.form.get("ncp")
    caec = request.form.get("caec")
    print(caec)
    smoke = request.form.get("smoke")
    ch2o = request.form.get("ch2o")
    scc = request.form.get("scc")
    faf = request.form.get("faf")
    tue = request.form.get("tue")
    calc = request.form.get("calc")
    mtrans = request.form.get("mtrans")
    print(mtrans)

    return {
        "data": {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "FamHist": fam,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }
    }

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/obesityRiskForm", methods=["GET", "POST"])
def predictform():
    if request.method == "POST":
        return redirect(url_for("predict"))
    return render_template("obesityRiskForm.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        sample = create_request_data(request=request)
        prediction = predict_sample(sample)
        return render_template("predict.html", prediction_class=prediction['class'])
    else:
        return render_template("predict.html", prediction="")


@app.post("/predict_api")
def predict_api():
    data = request.json
    try:
        sample = data["data"]
    except KeyError:
        return jsonify({"error": "Not text was fetched!"})
    result_data = predict_sample(sample)
    try:
        result = jsonify(result_data)
    except TypeError as e:
        result = jsonify({"error": str(e)})

    return result


if __name__ == "__main__":
    app.run(debug=True)