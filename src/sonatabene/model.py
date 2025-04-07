from ultralytics import YOLO
from numpy import ndarray
from torch import Tensor
from pathlib import Path

def train(data_path: str, model_path: str = "yolo11n.pt", **kwargs):
    """
    Train a YOLO model on a custom dataset.

    Args:
        data_path (str): Path to dataset configuration file (e.g. 'data/dataset.yaml')
        model_path (str, optional): Path to initial model weights.
        **kwargs: Additional training arguments including:
            epochs (int, optional): Number of training epochs.
            batch (int): Batch size (-1 for autobatch)
            imgsz (int): Input image size
            device (int|str): Device to run on (e.g. cuda device=0 or device=0,1,2,3 or device=cpu)
            workers (int): Number of worker threads for data loading (per RANK if DDP)
            optimizer (str): Optimizer to use (e.g. SGD, Adam, AdamW)
            patience (int): Epochs to wait for no observable improvement for early stopping

    For more training options, see:
    https://docs.ultralytics.com/fr/modes/train/#train-settings

    Returns:
        YOLO: Trained YOLO model instance
    """
    model = YOLO(model_path) 
    model.train(data=data_path, **kwargs)
    return model


def predict(image: str | Path | int | list | tuple | ndarray | Tensor = None, model_path: str = "models/yolo11n.pt", **kwargs):
    model = YOLO(model_path)
    return model.predict(image, **kwargs)