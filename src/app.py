"""
App module
"""


import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
import torch

from config import config
from dataset import get_transforms
from model import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Things to do on startup and shutdown of the app
    """
    
    # Load the trained model, set it to eval mode,
    # and attach it and the validation transforms to the app state for use in /predict endpoint
    model = get_model(load_pretrained=False)
    model.load_state_dict(torch.load(config["training"]["save_path"], map_location="cpu"))
    model.eval()

    app.state.model = model
    app.state.transform = get_transforms(is_training=False)

    print("Application startup complete. Model loaded and ready for predictions.")
    yield
    print("Shutting down the application...")


app = FastAPI(
    title="Logo Detector API",
    description="API for detecting logos in images",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """
    Return a simple status payload used for service health checks
    """

    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Run logo prediction on an uploaded image and return a binary label
    """

    image_bytes = await file.read()
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {str(e)}")
    
    tensor = app.state.transform(image).unsqueeze(0)
    with torch.no_grad():
        probability = torch.sigmoid(app.state.model(tensor).squeeze()).item()

    return {"prediction": 1 if probability > 0.5 else 0}