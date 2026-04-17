from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
import os
from classifier.RaceClassifier import RaceClassifier
from datetime import datetime
from validate import preprocess

# Initialize Flask app
app = Flask(__name__)

# Create folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Paths for DNN face detector
prototxt_path = "detectors/deploy.prototxt"
weights_path = "detectors/res10_300x300_ssd_iter_140000.caffemodel"

# Load DNN model
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# Load pre-trained classifier
model = RaceClassifier(model_path="fair_face_models/res34_fair_align_multi_7_20190809.pt")

# ------------------ Helper mappings ------------------
def map_ethnicity(idx):
    labels = ["White", "Black", "Latino_Hispanic", "East Asian",
              "Southeast Asian", "Indian", "Middle Eastern"]
    return labels[idx]

def map_gender(idx):
    return ["Male", "Female"][idx]

def map_age(idx):
    bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    return bins[idx]

# ------------------ Face detection ------------------
def detect_faces_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ------------------ Routes ------------------
@app.route('/')
def home():
    if preprocess()=="valid":
        return render_template('home.html')
    else:
        return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if preprocess() == "valid":
        if request.method == 'GET':
            return render_template('index.html')

        if 'file' not in request.files:
            return render_template('index.html', error="Please upload an image")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        # Read and decode uploaded image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template('index.html', error="Invalid image format")

        # Save uploaded image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(save_path, img)

        # Detect faces
        faces = detect_faces_dnn(img)
        if len(faces) == 0:
            return render_template('index.html', error="No faces detected in image")

        # Predict
        output = {}
        for i, (x, y, w, h) in enumerate(faces):
            roi = img[y:y+h, x:x+w]
            race, gender, age = model.predict(roi)
            output[f"Face {i+1}"] = {
                "ethnicity": map_ethnicity(race),
                "gender": map_gender(gender),
                "age": map_age(age)
            }

        if len(output) == 0:
            return render_template('index.html', error="Face detected but prediction failed")

        # Pass image path for display
        image_url = f"/{save_path.replace(os.sep, '/')}"
        return render_template('result.html', predictions=output, image_url=image_url)
    else:
        return render_template('base.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
