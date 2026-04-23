from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = MODELS_DIR / "email_dataset.csv"

MODEL_PATH = MODELS_DIR / "spam_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & (df["label"].isin(["safe", "spam", "scam"]))]
    return df


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = clean_text_column(df)

    if len(df) < 30:
        raise ValueError("Dataset is too small. Add more rows.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df["text"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="auto"
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(encoder.classes_),
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "labels": list(encoder.classes_),
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TRAINING COMPLETED ===")
    print(f"Dataset size    : {metrics['dataset_size']}")
    print(f"Train size      : {metrics['train_size']}")
    print(f"Test size       : {metrics['test_size']}")
    print(f"Accuracy        : {metrics['accuracy']}")
    print(f"Precision Macro : {metrics['precision_macro']}")
    print(f"Recall Macro    : {metrics['recall_macro']}")
    print(f"F1 Macro        : {metrics['f1_macro']}")
    print(f"Labels          : {metrics['labels']}")
    print(f"Saved model     : {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved encoder   : {ENCODER_PATH}")
    print(f"Saved metrics   : {METRICS_PATH}")


if __name__ == "__main__":
    main()