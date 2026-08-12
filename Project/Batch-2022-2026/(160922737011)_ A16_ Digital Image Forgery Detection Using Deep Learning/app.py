from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained CNN model
model = load_model('models/best_model_cnn.h5')

class_names = ['Fake', 'Real']
image_size = (128, 128)

# Allow TIFF format
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# ELA Processing
# ==============================
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image


def prepare_image(image_path):
    img = convert_to_ela_image(image_path, 90).resize(image_size)
    img = np.array(img).flatten() / 255.0
    img = img.reshape(-1, 128, 128, 3)
    return img


# ==============================
# ROUTES
# ==============================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/cnn-performance')
def cnn_performance():
    return render_template('cnn_performance.html')


@app.route('/mobilenet-performance')
def mobilenet_performance():
    return render_template('mobilenet_performance.html')


from werkzeug.utils import secure_filename

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        print("POST request received")

        if 'image' not in request.files:
            print("No image in request.files")
            return render_template('predict.html')

        file = request.files.get('image')

        if file.filename == '':
            print("No filename selected")
            return render_template('predict.html')

        if file and allowed_file(file.filename):
            print("File received:", file.filename)

            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print("File saved at:", filepath)

            # If TIFF, convert to JPG for display
            ext = filename.rsplit('.', 1)[1].lower()

            if ext in ['tif', 'tiff']:
                img = Image.open(filepath).convert("RGB")
                jpg_filename = filename.rsplit('.', 1)[0] + ".jpg"
                jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], jpg_filename)
                img.save(jpg_path, "JPEG")

                filename = jpg_filename  # Use JPG for rendering

            img = prepare_image(filepath)
            print("Image shape:", img.shape)

            y_pred = model.predict(img)
            print("Prediction raw:", y_pred)

            y_pred_class = np.argmax(y_pred, axis=1)[0]
            prediction = class_names[y_pred_class]
            confidence = float(np.max(y_pred)) * 100

            print("Prediction:", prediction, confidence)

    return render_template('predict.html',
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename)

if __name__ == '__main__':
    app.run(debug=True)