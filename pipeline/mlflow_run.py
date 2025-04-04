# pipeline/mlflow_run.py

import mlflow
import mlflow.pytorch
import yaml
import time
from ultralytics import YOLO

def train_yolo_from_config(config_path: str, base_model_path: str):
    """
    Read a YAML config file and train a YOLO model by recording
    the run in MLflow. Adjust according to your needs.
    """
    # Carga la config YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extraemos parámetros
    project = config.get("project", "runs/detect")
    epochs = config.get("epochs", 50)
    batch = config.get("batch", 16)
    imgsz = config.get("imgsz", 640)
    device = config.get("device", "0")
    workers = config.get("workers", 8)
    optimizer = config.get("optimizer", "auto")
    patience = config.get("patience", 10)

    data_yaml = "resources/data.yaml"  # Ajustez si votre data.yaml est dans un autre chemin

    with mlflow.start_run(run_name="Train_YOLO"):
        # We log parameters in MLflow
        mlflow.log_param("base_model_path", base_model_path)
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("project", project)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch", batch)
        mlflow.log_param("imgsz", imgsz)
        mlflow.log_param("device", device)
        mlflow.log_param("workers", workers)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("patience", patience)

        start = time.time()

        # Load YOLO template
        model = YOLO(base_model_path)

        # Training
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            project=project,
            imgsz=imgsz,
            device=device,
            workers=workers,
            optimizer=optimizer,
            patience=patience,
            name="finetuned_model"  # sous-dossier avec les résultats
        )

        end = time.time()
        mlflow.log_metric("training_time_sec", end - start)

        # In ultralytics >=8, "results" is an object that could have metrics
        # in results.metrics or results.run_cfg. Adjust according to your version:
        # Example (may vary):
        if hasattr(results, "metrics"):
            # "metrics" est parfois un dict avec "map50", "map" etc.
            metrics = results.metrics
            mlflow.log_metric("map50", metrics.get("box/map50", 0.0))
            mlflow.log_metric("map", metrics.get("box/map", 0.0))

        # Enregistrer le modèle entraîné (par exemple best.pt)
        # "results" crée un dossier avec "finetuned_model/weights/best.pt" etc.
        best_model_path = f"{project}/finetuned_model/weights/best.pt"

        if best_model_path:
            mlflow.log_artifact(best_model_path, artifact_path="model")
            print(f"[INFO] Model saved in: {best_model_path}")

        print("[INFO] Training completed.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO and track with MLflow using YAML config.")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--base_model", type=str, default="models/YOLO/yolo11m.pt",
                        help="Path of a YOLO base model to do finetuning.")
    args = parser.parse_args()

    train_yolo_from_config(args.config, args.base_model)

if __name__ == "__main__":
    main()
