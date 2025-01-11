ChestXrayCNN (Convolutional Neural Networks)
This project is a submission for the Model Minds Hackathon.
Made by shreeyuvan2020 (Shree Yuvan) and Samhith Pola

All imports below can be installed through "pip install -r requirements.txt", this will install all required dependencies for this project.

Imports:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import streamlit as st

Run "python -m streamlit run app.py" as this is opens the website and is where the inference is run, running the file normally will not work with Streamlit.
Here is the url for the hugging face dataset: https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset

THE DEVELOPERS OF THIS PROJECT ARE NOT LIABLE FOR ANY MALICIOUS AND OR UNINTENDED USE BY THE USER OF THIS PROJECT.


