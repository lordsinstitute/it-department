# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from speed_detector import process_video
from validate import preprocess

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXT = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'replace-with-a-secure-key'  # change for production

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if preprocess()=="valid":
        if request.method == 'POST':
            if 'video' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['video']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(in_path)

                # output filename
                out_name = os.path.splitext(filename)[0] + '_annotated.avi'
                out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

                try:
                    # process synchronously (may take time)
                    summary = process_video(in_path, out_path, haar_cascade_path='myhaar.xml')
                    return render_template('result.html', input_file=filename, output_file=out_name, summary=summary)
                except Exception as e:
                    flash(f'Error processing video: {e}')
                    return redirect(request.url)

        return render_template('upload.html')
    else:
        return render_template("index.html")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/outputs/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
