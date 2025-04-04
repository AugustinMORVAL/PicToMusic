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
    Load the image with PParser, detect and return the 'stafflines'
    """
    parser = PParser()
    image = parser.load_image(image_path)
    
    # Ajustez le paramètre selon vos critères ou laissez-le par défaut
    stafflines = parser.find_staff_lines(min_contour_area=5000)
    return parser, image, stafflines

def detect_notes_yolo(stafflines, model_path: str):
    """
    Applies a YOLO model to detect notes on each staffline.
    stafflines is the list of staves returned by PParser.
    Returns the YOLO detections for each staffline.
    """
    import numpy as np
    import cv2

    model = YOLO(model_path)
    detections = []

    for staffline in stafflines:
        # Convert grayscale to 3-channel BGR if needed
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


def run_pipeline(image_path: str, model_path: str, output_midi: str):
    """
    Run the end-to-end pipeline:
    1. Detect staves (PParser).
    2. YOLO for notes.
    3. Convert to MIDI with convert_to_midi.
    4. Log everything into MLflow.
    """
    # Start a run in MLflow (does not train, only records inference)
    with mlflow.start_run(run_name="pipeline_inference"):
        # Logueamos parámetros clave
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("output_midi", output_midi)

        start_time = time.time()

        # 1) Stave detection
        parser, image, stafflines = detect_staff_lines(image_path)
        mlflow.log_metric("num_stafflines", len(stafflines))

        # 2) Note detection with YOLO
        detections = detect_notes_yolo(stafflines, model_path)
        # Just as an example, it records how many stafflines we process
        mlflow.log_metric("num_stafflines_processed", len(detections))

        # 3) Converting to MIDI
        # Adjust whether your convert_to_midi function requires extra parameters
        convert_to_midi(detections, output_midi=output_midi)

        elapsed = time.time() - start_time
        mlflow.log_metric("inference_time_sec", elapsed)

        # 4) We upload the final MIDI as an artifact to the run
        if os.path.exists(output_midi):
            mlflow.log_artifact(output_midi)

        print(f"\n[INFO] Pipeline completed. MIDI saved to: {output_midi}")
        print(f"[INFO] Total time (seconds): {elapsed:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline end-to-end PicToMusic con MLflow.")
    parser.add_argument("--image", type=str, default="resources/samples/000051652-1_2_1.png",
                        help="Path of the score image to be processed.")
    parser.add_argument("--model", type=str, default="models/chopin.pt",
                        help="YOLO model path for note detection/classification.")
    parser.add_argument("--output_midi", type=str, default="output.mid",
                        help="Output path of the generated MIDI file.")

    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        model_path=args.model,
        output_midi=args.output_midi
    )

if __name__ == "__main__":
    main()
