import torch
import streamlit as st
from predict import pred_image, load_model
from src.model import make_model
from src.class_name import class_names

device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = make_model(class_names=class_names).to(device=device)
model_0.load_state_dict(torch.load("model/best_model.pth"))
model_1 = load_model(
    "model/best_model_v1,pth",
    model_fn=make_model,
    num_classes=class_names,
    device=device,
)



allowed_types = ["png", "jpg", "jpeg"]
st.title("🍛 Nepali Food Classifier")
st.subheader("Predictions")
upload_image = st.file_uploader("Upload Image", type=allowed_types)

if upload_image is not None:
    st.image(upload_image)
    # pred_image(image=upload_image, model=model, device=device)
