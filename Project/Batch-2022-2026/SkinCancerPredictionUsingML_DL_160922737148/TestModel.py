import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import joblib
def create_model():

    # Load dataset
    df = pd.read_csv("../Skin Cancer Dataset/hmnist_28_28_RGB.csv")
    X = df.drop(columns=["label"]).values / 255.0  # normalize
    y = df["label"].values

    # Balance dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluation
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save classification report
    report = classification_report(y_test, y_pred)
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 1.0, "Random Forest - Classification Report", fontsize=14, ha="center", weight="bold")
    plt.text(0.01, 0.98, report, fontsize=10, family="monospace", va="top")
    plt.axis("off")
    plt.savefig("rf_classification_report.png", bbox_inches="tight", dpi=300)
    plt.clf()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("rf_confusion_matrix.png", dpi=300)
    plt.clf()

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_score = clf.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Random Forest - Multiclass ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig("rf_roc_curve.png")
    plt.clf()


def test_model():
    classes = {
        0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),
        1: ('bcc', 'basal cell carcinoma'),
        2: ('bkl', 'benign keratosis-like lesions'),
        3: ('df', 'dermatofibroma'),
        4: ('nv', ' melanocytic nevi'),
        5: ('vasc', ' pyogenic granulomas and hemorrhage'),
        6: ('mel', 'melanoma'),
    }

    # Load trained model
    model = joblib.load(open("models/XGBoost_skin_model.pkl", "rb"))
    # Load image and convert to RGB
    img = Image.open("static/uploads/test.jpg").convert("RGB")

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert image to numpy array and flatten
    img_array = np.array(img).astype("float32") / 255.0  # normalize
    img_flat = img_array.reshape(1, -1)  # shape: (1, 2352)

    # Predict
    prediction = model.predict(img_flat)
    print(prediction[0])
    prediction_label=classes[prediction[0]]

    # Display result
    plt.imshow(img)
    plt.title(f"Prediction: {prediction_label}")
    plt.axis('off')
    plt.savefig('static/predictions/result.jpg')
    return prediction_label
"""
def test():
    classes = {
            0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),
            1: ('bcc', 'basal cell carcinoma'),

            2: ('bkl', 'benign keratosis-like lesions'),

            3: ('df', 'dermatofibroma'),

            4: ('nv', ' melanocytic nevi'),

            5: ('vasc', ' pyogenic granulomas and hemorrhage'),

            6: ('mel', 'melanoma'),
        }

    # Load trained model
    model = joblib.load(open("../models/XGBoost_skin_model.pkl", "rb"))

    # Predict
    predicted_class = test_random_forest_model("../img.png", model)
    print(predicted_class)
    print(f"Predicted Class: {classes[predicted_class]}")

    return predicted_class
"""