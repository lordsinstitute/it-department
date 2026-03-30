import os
import cv2
import numpy as np
from face_utils import train_model, load_labels
import pickle
from datetime import datetime
from database import get_connection
from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from admin import admin_bp
import face_recognition


# Define models directories
DATASET_DIR = os.path.join('dataset')
MODEL_DIR = os.path.join('models')
ENCODINGS_PATH = os.path.join(MODEL_DIR, 'encodings.pkl')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.pkl')

CASCADE_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
DATASET_PATH = os.path.join(os.getcwd(), "dataset")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#admin_bp = Blueprint('admin', __name__, template_folder='../templates/admin')

@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # ✅ simple demo logic, replace with real auth check
        if username == 'admin' and password == 'admin123':
            flash('Login successful', 'success')
            #return redirect(url_for('admin.dashboard'))
            return redirect(url_for('admin.admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('admin_login.html')

@admin_bp.route('/dashboard')
def admin_dashboard():
    """
    Admin dashboard showing total students and last trained date.
    """
    # 1️⃣ Count total students
    if os.path.exists(DATASET_DIR):
        total_students = len([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])
    else:
        total_students = 0

    # 2️⃣ Get last trained date
    if os.path.exists(ENCODINGS_PATH):
        last_trained = datetime.fromtimestamp(
            os.path.getmtime(ENCODINGS_PATH)
        ).strftime('%d-%b-%Y %H:%M')
    else:
        last_trained = "Not Trained Yet"

    return render_template(
        'admin_dashboard.html',
        total_students=total_students,
        last_trained=last_trained
    )

@admin_bp.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')

        if not student_id or not name:
            flash('All fields are required!', 'danger')
            return redirect(url_for('admin.add_student'))

        # Create dataset folder for student
        student_folder = os.path.join(DATASET_DIR, student_id)
        os.makedirs(student_folder, exist_ok=True)

        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        cap = cv2.VideoCapture(0)

        img_count = 0
        flash("Starting camera... Press 'q' to quit early.", "info")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                img_count += 1
                face = frame[y:y + h, x:x + w]
                face_path = os.path.join(student_folder, f"{student_id}_{img_count}.jpg")
                cv2.imwrite(face_path, face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Capturing {img_count}/30", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2)

            cv2.imshow("Capturing Faces - Press Q to stop", frame)

            # Stop after capturing 30 images or user presses Q
            if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= 30:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save to database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO students (student_id, name, images_captured) VALUES (?, ?, ?)",
                       (student_id, name, img_count))
        conn.commit()
        conn.close()

        flash(f"Student {name} registered successfully with {img_count} images!", 'success')
        return redirect(url_for('admin.add_student'))

    return render_template('add_student.html')


# -----------------------------------------------------------
# 1. Capture images for a new student (Admin Module)
# -----------------------------------------------------------
def capture_student_images(student_name, num_images=20):
    """
    Capture face images for a new student using webcam.
    Saves cropped face images into dataset/{student_name}/.
    """
    student_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    print(f"[INFO] Capturing images for {student_name}...")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(student_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}/{num_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {count} images saved to {student_dir}")


# -----------------------------------------------------------
# 2. Train Model on Collected Images
# -----------------------------------------------------------
@admin_bp.route('/train_model', methods=['GET', 'POST'])
def train_model():
    """
    Train the face recognition models using the face_recognition library.
    Encodes all images in dataset/ and saves encodings + labels to pickle files.
    """
    known_encodings = []
    known_labels = []

    print("[INFO] Training models on student dataset...")

    for student_name in os.listdir(DATASET_PATH):
        student_path = os.path.join(DATASET_PATH, student_name)
        if not os.path.isdir(student_path):
            continue

        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, face_locations)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_labels.append(student_name)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")

    # Save encodings and labels to pickle files
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(known_encodings, f)

    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(known_labels, f)

    print(f"[INFO] Training complete. {len(known_labels)} faces encoded.")
    flash(f"Training complete! {len(known_labels)} faces encoded successfully.", "success")

    return redirect(url_for('admin.admin_login'))



# -----------------------------------------------------------
# 3. Load Trained Encodings
# -----------------------------------------------------------
def load_labels():
    """
    Load known encodings and labels for face recognition.
    Returns: (known_encodings, known_labels)
    """
    if not os.path.exists(ENCODINGS_PATH) or not os.path.exists(LABELS_PATH):
        print("[WARN] Encodings or labels not found. Please train the models first.")
        return [], []

    with open(ENCODINGS_PATH, 'rb') as f:
        known_encodings = pickle.load(f)

    with open(LABELS_PATH, 'rb') as f:
        known_labels = pickle.load(f)

    return known_encodings, known_labels


# -----------------------------------------------------------
# 4. Mark Attendance (User Module)
# -----------------------------------------------------------
def mark_attendance(student_name):
    """
    Records student attendance in SQLite database.
    """
    conn = get_connection()
    cur = conn.cursor()

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    # Check if already marked
    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id = (SELECT id FROM students WHERE name = ?)
        AND date = ?
    """, (student_name, date_str))
    record = cur.fetchone()

    if record:
        print(f"[INFO] Attendance already marked for {student_name} on {date_str}")
    else:
        cur.execute("""
            INSERT INTO attendance (student_id, date, time, status)
            VALUES ((SELECT id FROM students WHERE name = ?), ?, ?, ?)
        """, (student_name, date_str, time_str, "Present"))
        conn.commit()
        print(f"[INFO] Attendance marked for {student_name}.")

    conn.close()


# -----------------------------------------------------------
# 5. Recognize Faces in Real-Time
# -----------------------------------------------------------
def recognize_and_mark_attendance():
    """
    Starts webcam and recognizes students in real-time.
    Marks attendance for recognized students.
    """
    known_encodings, known_labels = load_labels()
    if not known_encodings:
        print("[ERROR] No trained models found. Please train the models first.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_loc = face_recognition.face_locations(rgb_small)
        faces_enc = face_recognition.face_encodings(rgb_small, faces_loc)

        for (top, right, bottom, left), face_encoding in zip(faces_loc, faces_enc):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_dist = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"

            if len(face_dist) > 0:
                best_match = np.argmin(face_dist)
                if matches[best_match]:
                    name = known_labels[best_match]

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Real-time recognition stopped.")
