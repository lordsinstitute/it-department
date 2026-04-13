# face_utils_recog.py
import cv2
import numpy as np
import face_recognition
import pickle
import os

MODEL_DIR = os.path.join(os.getcwd(), "models")
ENCODINGS_PATH = os.path.join(MODEL_DIR, "encodings.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

def recognize_faces(frame):
    """Recognize faces in the given frame using trained encodings."""
    if not os.path.exists(ENCODINGS_PATH) or not os.path.exists(LABELS_PATH):
        print("[ERROR] Trained model not found. Please train using admin module.")
        return None

    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings = pickle.load(f)

    with open(LABELS_PATH, "rb") as f:
        known_labels = pickle.load(f)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_name = None

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                recognized_name = known_labels[best_match_index]
                break

    return recognized_name
