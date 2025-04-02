#!/usr/bin/env python3

"""
Script: split_dataset.py
------------------------
Splits the annotated images (all in 'images/all') and their .txt files (in 'labels/all')
into train/val/test/ files randomly (e.g., 80/10/10).
"""

import os
import shutil
import random

IMAGES_DIR = "resources/dataset_yolo/images/all"
LABELS_DIR = "resources/dataset_yolo/labels/all"

TRAIN_IMG_DIR = "resources/dataset_yolo/images/train"
VAL_IMG_DIR   = "resources/dataset_yolo/images/val"
TEST_IMG_DIR  = "resources/dataset_yolo/images/test"

TRAIN_LBL_DIR = "resources/dataset_yolo/labels/train"
VAL_LBL_DIR   = "resources/dataset_yolo/labels/val"
TEST_LBL_DIR  = "resources/dataset_yolo/labels/test"

TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1

def main():
    # Create folders if they don't exist
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
              TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR]:
        os.makedirs(d, exist_ok=True)

    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n_total = len(images)
    train_count = int(TRAIN_SPLIT * n_total)
    val_count   = int(VAL_SPLIT * n_total)

    train_images = images[:train_count]
    val_images   = images[train_count:train_count+val_count]
    test_images  = images[train_count+val_count:]

    # Function to move image and label
    def move_image_and_label(img_list, img_dest, lbl_dest):
        for img in img_list:
            base, _ = os.path.splitext(img)
            label_file = base + ".txt"

            # Move image
            src_img = os.path.join(IMAGES_DIR, img)
            dst_img = os.path.join(img_dest, img)
            shutil.move(src_img, dst_img)

            # Move label (if exists)
            src_lbl = os.path.join(LABELS_DIR, label_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(lbl_dest, label_file)
                shutil.move(src_lbl, dst_lbl)

    move_image_and_label(train_images, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    move_image_and_label(val_images,   VAL_IMG_DIR,   VAL_LBL_DIR)
    move_image_and_label(test_images,  TEST_IMG_DIR,  TEST_LBL_DIR)

    print(f"Total images: {n_total}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")

if __name__ == "__main__":
    main()
