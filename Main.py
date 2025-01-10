import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.optim.lr_schedulers import StepLR
# Create the environment


# Set device
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Define transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Preprocessing function for dataset
def preprocess(example):
    image_data = example["image"]
    if not isinstance(image_data, Image.Image):
        image_data = Image.fromarray(image_data)
    example["image"] = transform(image_data)
    return example

# Load and preprocess dataset
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification")
dataset = dataset.map(preprocess)
# Split dataset into train, validation, and test sets
train_data = dataset["train"]
split_index = int(0.8 * len(train_data))
train_dataset = train_data.select(range(split_index))
val_dataset = train_data.select(range(split_index, len(train_data)))
test_dataset = dataset["test"]

# Collate function for DataLoader
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return images, labels

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the CNN
class ChestCNN(nn.Module):
    def __init__(self):
        super(ChestCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Conv1: 1x64x64 -> 16x64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Conv2: 16x32x32 -> 32x32x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Conv3: 32x16x16 -> 64x16x16
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Conv4: 64x8x8 -> 128x8x8
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Pooling: Downsampling by factor of 2
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Fully connected: 64x8x8 -> 256
        self.fc2 = nn.Linear(256, 14)  # Fully connected: 256 -> 10 classes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pool: 16x64x64 -> 16x32x32
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pool: 32x32x32 -> 32x16x16
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3 + ReLU + Pool: 64x16x16 -> 64x8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = torch.relu(self.fc1(x))  # FC1 + ReLU
        x = self.dropout(x)         # Dropout
        x = self.fc2(x)             # FC2
        return x

# Initialize model, loss function, and optimizer
model = ChestCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training function
def train_model(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Validation function
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    train_acc = evaluate_model(model, train_loader)
    val_acc = evaluate_model(model, val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")
    scheduler.step()

# Final test accuracy
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
model = torch.save(model.state_dict(), "C:\Users\yuva\workspace\Main.py\ChestCNN.pth")

