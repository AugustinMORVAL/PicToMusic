from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from fastapi.responses import Response
import pickle

app = FastAPI()
model = YOLO("models/chopin.pt")

@app.get("/")
def root():
    return {"message": "API YOLO prête à prédire !"}

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Read the image content
        contents = await file.read()

        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)

        # Decode image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Sanity check
        if image is None:
            raise ValueError("Could not decode the image.")

        # Run YOLO model prediction
        results = model(image)

        result_bytes = pickle.dumps(results)

        return Response(content=result_bytes, media_type="application/octet-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})