
from matplotlib.pyplot import cla
import torch
import streamlit as st
from predict import pred_image, load_model
from src.model import make_model
from src.class_name import class_names
from PIL import Image
from pathlib import Path
import pandas as pd
from Grad_CAM_Visualization import generate_gradcam


st.set_page_config(page_title="Nepali Food Classification", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_0, cnn = load_model(
    "model/best_model.pth", model_fn=make_model, num_classes=class_names, device=device
)
model_1, cn = load_model(
    "model/best_model_v1,pth",
    model_fn=make_model,
    num_classes=class_names,
    device=device,
)


def make_pred(img,img_dir ,example=False):
    col1, col2 = st.columns(2)
    if not example:
        img = Image.open(img).convert("RGB")


    with col1:
        st.subheader("Model 0 (Basic data Augmentation ResNet18)")
        model_0_pred = pred_image(
            image=img, model=model_0, class_names=cn, device=device
        )
        model_0_pred_df = pd.DataFrame(model_0_pred)
        st.image(
            generate_gradcam(model=model_0, image_path=img_dir, device=device)
        )

        for i in range(len(model_0_pred_df)):
            st.write(
                f"{model_0_pred_df['class'][i]}  ---  {model_0_pred_df['Confidance'][i]}"
            )
        model_0_pred_df["Confidance"] = pd.to_numeric(model_0_pred_df["Confidance"])
        st.bar_chart(model_0_pred_df, x="class", y="Confidance", sort=True)
        st.write(Image.open("model/model_0.png"))
    with col2:
        st.subheader("Model 1 (Advanced Augmentation with scheduler on ResNet18)")
        model_1_pred = pred_image(
            image=img, model=model_1, class_names=cn, device=device
        )
        model_1_pred_df = pd.DataFrame(model_1_pred)
        st.image(
            generate_gradcam(model=model_1, image_path=img_dir, device=device)
        )
        for i in range(len(model_1_pred_df)):
            st.write(
                f"{model_1_pred_df['class'][i]}  ---  {model_1_pred_df['Confidance'][i]}"
            )
        model_1_pred_df["Confidance"] = pd.to_numeric(model_1_pred_df["Confidance"])
        st.bar_chart(model_1_pred_df, x="class", y="Confidance", sort=True)
        st.write(Image.open("model/model_1.png"))


c1, c2 = st.columns(2)
with c1:
    allowed_types = ["png", "jpg", "jpeg"]
    st.title("🍛 Nepali Food Classifier")
    st.subheader("Predictions")
    upload_image = st.file_uploader("Upload Image", type=allowed_types)

if upload_image is not None:
    st.image(upload_image, width=224)
    with st.spinner("Analysing...", show_time=True):
        make_pred(img=upload_image,img_dir=upload_image)
else:
    example_dir = Path("src/streamlit_app/example.jpg")
    upload_image = Image.open(example_dir).convert("RGB")
    if example_dir.is_file():
        st.write("Example Image:")
        st.image(upload_image,width=244)
        with st.spinner("Analysing...", show_time=True):
            make_pred(img=upload_image,img_dir=example_dir,example=True)


st.sidebar.title("About")

st.sidebar.write(
    "ResNet18 trained On : chatpate, dal_bhat, gundruk, momo, samosa, sel_roti, yomari"
)
