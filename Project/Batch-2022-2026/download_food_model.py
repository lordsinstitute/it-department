import os
from huggingface_hub import snapshot_download
from transformers import pipeline
from PIL import Image

# -----------------------------------
# Step 1: Download Model
# -----------------------------------

MODEL_NAME = "nateraw/food"
LOCAL_DIR = "./food_model"
"""
print("Downloading model...")

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
)

print("Model downloaded successfully!\n")
"""
# -----------------------------------
# Step 2: Load Model from Local Folder
# -----------------------------------

print("Loading model from local directory...")

classifier = pipeline(
    "image-classification",
    model=LOCAL_DIR
)

print("Model loaded successfully!\n")

# -----------------------------------
# Step 3: Test Prediction (Optional)
# -----------------------------------

# Replace with your test image
TEST_IMAGE = "D:/Projects 25-26/Source codes/Datasets/food_101/images/cheese_plate/128927.jpg"

if os.path.exists(TEST_IMAGE):
    result = classifier(TEST_IMAGE)
    print("Prediction:")
    print(result[0])
else:
    print("Add a test image named 'test_food.jpg' to test prediction.")
