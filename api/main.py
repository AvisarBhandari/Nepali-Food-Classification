from fastapi import FastAPI, UploadFile
from PIL import Image
from matplotlib.pyplot import cla
import torch
from src.model import make_model
from src.streamlit_app.predict import load_model,pred_image
import sys
from pathlib import Path
from src.class_name import class_names


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model,class_name = load_model("model/best_model.pth",make_model,class_names,device=device)

@app.get("/")
def root():
    return {"device": device,
            "class name":class_names}
@app.post("/predict")
async def predict(file:UploadFile):
    img = Image.open(file.file).convert("RGB")
    pred = pred_image(image=img,model=model,class_names=class_name,device=device)
    return pred