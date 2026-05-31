from fastapi import FastAPI, UploadFile
from PIL import Image
from matplotlib.pyplot import cla
import torch
from src.streamlit_app.predict import load_model, pred_image
from src.model import make_model
import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class_names = 7
device = "cuda" if torch.cuda.is_available() else "cpu"
model, class_name = load_model(
    "model/best_model.pth", make_model, class_names, device=device
)


@app.get("/")
def root():
    return {"device": device, "class name": class_names}


@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(file.file).convert("RGB")

    predictions = pred_image(
        image=img,
        model=model,
        class_names=class_name,
        device=device
    )

    return {
        "prediction": predictions[0]["class"],
        "confidence": float(predictions[0]["Confidance"]),
        "top_predictions": predictions
    }