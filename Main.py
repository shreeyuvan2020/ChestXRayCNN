import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
def preprocess(batch):
    processed_images = []
    for image_data in batch["image"]:
        image_data = np.squeeze(image_data)

        if len(image_data.shape) == 2:
            image_data = np.stack([image_data] * 3, axis=-1)
        elif len(image_data.shape) == 3 and image_data.shape[-1] == 1:
            image_data = np.repeat(image_data, 3, axis=-1)

        if np.issubdtype(image_data.dtype, np.floating):
            image_data = (image_data * 255).clip(0, 255).astype(np.uint8)

        # Step 4: Convert to PIL Image
        pil_image = Image.fromarray(image_data, mode="RGB")
        pil_image = pil_image.resize((64, 64))
        transformed_image = transform(pil_image)

        processed_images.append(transformed_image)

    batch["image"] = processed_images
    return batch

dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification")
dataset = dataset.map(preprocess, batched=True)

def encode_label(label, num_classes):
    one_hot = torch.zeros(num_classes, dtype=torch.float32)
    for lbl in label:
        one_hot[lbl-1] = 1
    return one_hot

train_data = dataset["train"]
split_index = int(0.8 * len(train_data))
train_dataset = train_data.select(range(split_index))
val_dataset = train_data.select(range(split_index, len(train_data)))
test_dataset = dataset["test"]

def collate_fn(batch):
    images = torch.stack([torch.tensor(item["image"]) for item in batch])
    labels = torch.stack([encode_label(item["labels"], 15) for item in batch]).float()
    return images, labels

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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

model = ChestCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
saved_dataset = dataset.map(preprocess, batched=True)
saved_dataset.save_to_disk("chest_xray_dataset") 
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

def evaluate_model(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")
    return f1

# Training loop
num_epochs = 5
best_f1 = 0.0
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    train_acc = evaluate_model(model, train_loader)
    val_acc = evaluate_model(model, val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")
    scheduler.step(-val_acc)
    if val_acc > best_f1:
        best_f1 = val_acc
        torch.save(model.state_dict(), "best_model.pth")

# Final test accuracy
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")

