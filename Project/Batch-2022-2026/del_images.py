import os
import random
import shutil

# Path to the parent directory containing all 100 folders
parent_dir = "images"

# Loop through each folder
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)

    if os.path.isdir(folder_path):
        # Get all image files in the folder
        images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Randomly select 50 images to keep
        keep_images = set(random.sample(images, 50))

        # Delete the rest
        for img in images:
            if img not in keep_images:
                os.remove(os.path.join(folder_path, img))

print("Done! Each folder now has only 50 images.")
