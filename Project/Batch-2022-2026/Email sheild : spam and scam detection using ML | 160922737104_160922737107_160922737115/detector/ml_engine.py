import json
import joblib

from .config import Config

MODEL_PATH = Config.MODELS_DIR / "spam_model.joblib"
VECTORIZER_PATH = Config.MODELS_DIR / "vectorizer.joblib"
ENCODER_PATH = Config.MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = Config.MODELS_DIR / "metrics.json"


def model_exists() -> bool:
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists() and ENCODER_PATH.exists()


def load_model_artifacts():
    if not model_exists():
        raise FileNotFoundError("Trained model not found. Please run: python train_model.py")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, vectorizer, encoder


def load_metrics():
    if not METRICS_PATH.exists():
        return {
            "dataset_size": 0,
            "train_size": 0,
            "test_size": 0,
            "labels": [],
            "accuracy": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "f1_macro": 0,
            "confusion_matrix": []
        }

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "dataset_size": 0,
            "train_size": 0,
            "test_size": 0,
            "labels": [],
            "accuracy": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "f1_macro": 0,
            "confusion_matrix": []
        }