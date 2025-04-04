# pipeline/pipeline.py

import argparse
import os
import time
from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.p2m.parser import PParser
from src.p2m.converter2 import convert_to_midi

def detect_staff_lines(image_path: str):
    """
    Detects staff lines from the sheet music image using PParser.
    """
    parser = PParser()
    image = parser.load_image(image_path)
    stafflines = parser.find_staff_lines(min_contour_area=5000)
    return parser, image, stafflines

def detect_notes_yolo(stafflines, model_path: str):
    """
    Uses a YOLO model to detect notes on each staffline image.
    """
    import numpy as np
    import cv2

    model = YOLO(model_path)
    detections = []

    for staffline in stafflines:
        if len(staffline.image.shape) == 2:
            image_bgr = cv2.cvtColor(staffline.image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = staffline.image

        results = model.predict(
            source=image_bgr,
            verbose=False
        )
        detections.append(results)
    return detections

def run_pipeline(image_path: str, model_notes_path: str, model_symbols_path: str, output_midi: str):
    """
    End-to-end pipeline for sheet music to MIDI conversion:
    1. Staff line detection (PParser)
    2. Note detection (YOLO, chopin.pt)
    3. Symbol detection (YOLO, bach.pt) – future integration
    4. Conversion to MIDI
    5. MLflow logging
    """
    with mlflow.start_run(run_name="pipeline_inference"):
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("model_notes_path", model_notes_path)
        mlflow.log_param("model_symbols_path", model_symbols_path)
        mlflow.log_param("output_midi", output_midi)

        start_time = time.time()

        # Step 1: Detect staff lines
        parser, image, stafflines = detect_staff_lines(image_path)
        mlflow.log_metric("num_stafflines", len(stafflines))

        # Step 2: Detect notes with chopin.pt
        detections = detect_notes_yolo(stafflines, model_notes_path)
        mlflow.log_metric("num_stafflines_processed", len(detections))

        # Step 3: (optional future) detect other symbols with bach.pt

        # Step 4: Convert detections to MIDI
        convert_to_midi(detections, output_midi=output_midi)

        elapsed = time.time() - start_time
        mlflow.log_metric("inference_time_sec", elapsed)

        # Step 5: Log final MIDI file
        if os.path.exists(output_midi):
            mlflow.log_artifact(output_midi)

        print(f"\n[INFO] Pipeline completed. MIDI saved to: {output_midi}")
        print(f"[INFO] Total time (seconds): {elapsed:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run PicToMusic pipeline with MLflow.")
    parser.add_argument("--image", type=str, default="resources/samples/000051652-1_2_1.png")
    parser.add_argument("--model_notes", type=str, default="models/chopin.pt")
    parser.add_argument("--model_symbols", type=str, default="models/bach.pt")
    parser.add_argument("--output_midi", type=str, default="output.mid")

    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        model_notes_path=args.model_notes,
        model_symbols_path=args.model_symbols,
        output_midi=args.output_midi
    )

if __name__ == "__main__":
    main()
