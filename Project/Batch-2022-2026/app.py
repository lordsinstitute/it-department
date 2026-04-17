import os
from flask import *
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from views import app_exec_check
# ---- Configuration ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_WEIGHTS_DIR = os.path.join("model", "weights")

# Expected class labels
PART_LABELS = ['Elbow', 'Hand', 'Shoulder']
FRACTURE_LABELS = ['fractured', 'normal']

# ---- App init ----

app = Flask(__name__)
app_exec_check(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "abc"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---- Helper functions ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    part_model_path = os.path.join(MODEL_WEIGHTS_DIR, "ResNet50_BodyParts.h5")

    if not os.path.exists(part_model_path):
        raise FileNotFoundError(f"Body parts model not found at {part_model_path}")

    print("Loading body-parts model ...")
    body_part_model = tf.keras.models.load_model(part_model_path)

    fracture_models = {}
    for p in PART_LABELS:
        mpath = os.path.join(MODEL_WEIGHTS_DIR, f"ResNet50_{p}_frac.h5")
        if not os.path.exists(mpath):
            print(f"Warning: fracture model for {p} not found.")
            fracture_models[p] = None
        else:
            print(f"Loading fracture model for {p} ...")
            fracture_models[p] = tf.keras.models.load_model(mpath)

    return body_part_model, fracture_models


# Load models
body_part_model, fracture_models = load_models()


# ----------- Utility to list images -----------
def get_images(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_reports(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
# ----------- Pages -----------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/part-model")
def part_model_page():
    folder = os.path.join("static", "plots")
    images = get_images(folder)

    reports = get_reports(folder)

    report_content = None
    if reports:
        with open(os.path.join(folder, reports[0]), "r") as f:
            report_content = f.read()

    return render_template("model_details.html",
                           title="Body Part Model Details",
                           folder="plots",
                           images=images,
                           report=report_content)


@app.route("/elbow")
def elbow_page():
    folder = os.path.join("static", "plots/FractureDetection/Elbow")
    images = get_images(folder)

    reports = get_reports(folder)

    report_content = None
    if reports:
        with open(os.path.join(folder, reports[0]), "r") as f:
            report_content = f.read()

    return render_template("model_details.html",
                           title="Elbow Model",
                           folder="plots/FractureDetection/Elbow",
                           images=images,
                           report=report_content)


@app.route("/hand")
def hand_page():
    folder = os.path.join("static", "plots/FractureDetection/Hand")
    images = get_images(folder)
    reports = get_reports(folder)

    report_content = None
    if reports:
        with open(os.path.join(folder, reports[0]), "r") as f:
            report_content = f.read()

    return render_template("model_details.html",
                           title="Hand Model",
                           folder="plots/FractureDetection/Hand",
                           images=images,
                           report=report_content)


@app.route("/shoulder")
def shoulder_page():
    folder = os.path.join("static", "plots/FractureDetection/Shoulder")
    images = get_images(folder)
    reports = get_reports(folder)

    report_content = None
    if reports:
        with open(os.path.join(folder, reports[0]), "r") as f:
            report_content = f.read()

    return render_template("model_details.html",
                           title="Shoulder Model",
                           folder="plots/FractureDetection/Shoulder",
                           images=images,
                           report=report_content)


# ----------- Predict Page -----------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # File validation
        if 'file' not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # ---- Preprocess ----
            x = preprocess_image(save_path)

            # ---- Predict body part ----
            part_preds = body_part_model.predict(x)
            part_idx = int(np.argmax(part_preds))
            predicted_part = PART_LABELS[part_idx]
            part_conf = float(np.max(tf.nn.softmax(part_preds).numpy()))

            # ---- Predict fracture ----
            fracture_model = fracture_models.get(predicted_part)

            if fracture_model is None:
                return render_template("result1.html",
                                       image_url=url_for('uploaded_file', filename=filename),
                                       predicted_part=predicted_part,
                                       part_confidence=round(part_conf * 100, 2),
                                       fracture_prediction="Not Available",
                                       fracture_confidence=0,
                                       error="Model not available")

            y = preprocess_image(save_path)
            frac_preds = fracture_model.predict(y)
            frac_idx = int(np.argmax(frac_preds))
            predicted_frac = FRACTURE_LABELS[frac_idx]
            frac_conf = float(np.max(tf.nn.softmax(frac_preds).numpy()))

            return render_template("result1.html",
                                   image_url=url_for('uploaded_file', filename=filename),
                                   predicted_part=predicted_part,
                                   part_confidence=round(part_conf * 100, 2),
                                   fracture_prediction=predicted_frac,
                                   fracture_confidence=round(frac_conf * 100, 2),
                                   error=None)

        else:
            flash("Invalid file format")
            return redirect(request.url)

    return render_template("predict.html")


# ----------- Image Upload Serve -----------

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ----------- Preprocessing -----------

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(arr)


# ----------- Health Check -----------

@app.route("/health")
def health():
    return {"status": "ok"}


# ----------- Run App -----------

if __name__ == "__main__":
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    app.run(debug=True)