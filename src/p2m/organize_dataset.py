#!/usr/bin/env python3

"""
Script: organize_dataset.py
---------------------------
1. Scan the subfolders of resources/output/parsing_test/
2. Find image files (JPG, PNG)
3. Copy all valid images to resources/dataset_yolo/images/all/
(Create an 'all' folder to keep them together)
4. Print the total number of images copied.
"""

import os
import shutil

SOURCE_DIR = "resources/output/parsing_test"
TARGET_DIR = "resources/dataset_yolo/images/all"

def main():
    # Create folder 'all' if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)

    count = 0
    # We recursively traverse the folders
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # Filter only images
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                source_path = os.path.join(root, file)
                target_path = os.path.join(TARGET_DIR, file)

                # Avoid collisions: rename if exists
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    # Add a suffix to avoid stepping on it
                    new_file = f"{base}_dup{count}{ext}"
                    target_path = os.path.join(TARGET_DIR, new_file)

                shutil.copy2(source_path, target_path)
                count += 1

    print(f"We've copied the number {count} images to the {TARGET_DIR}")

if __name__ == "__main__":
    main()
