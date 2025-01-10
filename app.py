from Main.py import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import streamlit as st
from Main.py import transform
from Main.py import device

st.title("Chest X-Ray Classification")
uploaded_file = st.file_uploader("Choose an image...", type="png")
submit_button = st.button("Submit")
if submit_button and uploaded_file is not None:
    image = Image.open(uploaded_file)
    transformed_image = transform(image).unsqueeze(0).to(device)  # Prepare image for prediction
    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write(f"Predicted Label: {predicted.item()}")