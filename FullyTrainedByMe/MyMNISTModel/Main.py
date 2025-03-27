import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt


# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install matplotlib==3.9.0
# version < 3.10 because 3.10 has an issue when plotting images

def save_model():
    # Save model
    torch.save(model.state_dict(), "number_classifier.pth")
    
def load_model():
    # Load model
    model.load_state_dict(torch.load("number_classifier.pth"))
    model.eval()  # Set to evaluation mode


# Define transformations: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

# Download and load test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)


class NumberClassifier(nn.Module):
    def __init__(self):
        super(NumberClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = NumberClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


load_model()
def train_model():
    # Training loop
    epochs = 10  # Number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
    
            running_loss += loss.item()
    
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
        save_model()
    
    print("Training complete!")
    
    
# train_model()
save_model()
model.eval()


# testing
def test():
    correct = 0
    total = 0
    with torch.no_grad(): # No gradients needed (faster computation)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Get class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')


test()

def predict_wrong_once():
    # Get a sample image
    image, label = testset[random.randint(0, len(testset)-1)]
    # Predict
    new_image = image.unsqueeze(0).to(device)
    output = model(new_image)
    _, predicted = torch.max(output, 1)
    
    if predicted != label:
        print("Wrong!")
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.show(block=True)
        
        print(f'Predicted Label: {predicted.item()}, True Label: {label}')
    
def find_wrong_predictions():
    for _ in range(1000):
        predict_wrong_once()


find_wrong_predictions()

def predict_one_img():
    # Get a sample image
    image, label = testset[random.randint(0, len(testset) - 1)]
    # Predict
    new_image = image.unsqueeze(0).to(device)
    output = model(new_image)
    _, predicted = torch.max(output, 1)
    
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.show(block=True)
    
    print(f'Predicted Label: {predicted.item()}, True Label: {label}')
    

# for _ in range(3):
#     predict_one_img()
    