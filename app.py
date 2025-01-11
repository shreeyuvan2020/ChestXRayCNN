import streamlit as st
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_from_disk
import numpy as np
from test.py import transform
from torch.optim.lr_scheduler import StepLR

st.title("Chest X-Ray Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
submit_button = st.form_submit_button("Submit")
class ChestCNN(nn.Module):
    def __init__(self):
        super(ChestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 15)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x))) 
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ChestCNN()
torch.load("best_model.pth")
model.load_state_dict("best_model.pth")
if submit_button and uploaded_file is not None:
    with torch.no_grad():
        image = Image.open(uploaded_file)
        image = transform(image)
        image = image.unsqueeze(0)
        model.eval()
        raw_output = model(image)
        probabilities = torch.sigmoid(raw_output)
        predicted = (probabilities > 0.5).float()
        labels = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
    "Pneumonia",
    "Hernia"]
        predicted_labels = [labels[idx] for idx, val in enumerate(predicted[0]) if val == 1]

        st.image(image, caption="Uploaded Image.", use_column_width=True)
        font_size = 70
        if predicted_labels:
            conditions = ", ".join(predicted_labels)
            st.markdown(f"<h1 style='font-size:50px;'>You most likely have: {conditions}</h1>", unsafe_allow_html=True)
    
