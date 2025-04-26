import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Train the model on the MNIST dataset
def train_model(epoch_range):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed to Adam optimizer

    model.train()
    for epoch in range(epoch_range):  # Train for the specified number of epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_cnn.pt')
    print("Model trained and saved as mnist_cnn.pt")

# Function to evaluate the model on the test dataset and calculate accuracy
def evaluate_model(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

# Function to preprocess the input image and make predictions
def predict_digit(image_path, model):
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img, dtype=np.float32)
    img = (255 - img) / 255.0  # Invert and normalize
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Debugging: Check the shape and values of the image
    print(f"Input image shape: {img.shape}")
    print(f"Input image: {img.numpy()}")

    # Make the prediction
    with torch.no_grad():
        output = model(img)
        prediction = output.argmax(dim=1, keepdim=True).item()

    return prediction

# Create the GUI
def upload_image(model):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_digit = predict_digit(file_path, model)
        result_label.config(text=f"The predicted digit is: {predicted_digit}")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("400x200")

# Specify the epoch range
epoch_range = 5  # Change this value to retrain the model

# Check if the model file exists
model = Net()
model_file = 'mnist_cnn.pt'
if os.path.exists(model_file):
    # Load existing model weights
    model.load_state_dict(torch.load(model_file))
    print("Loaded existing model weights.")
else:
    # Train the model if no existing model
    train_model(epoch_range)

# Evaluate the model on the test dataset to check accuracy
evaluate_model(model)

# Create upload button and result label
upload_button = Button(root, text="Upload Image", command=lambda: upload_image(model))
upload_button.pack(pady=20)

result_label = Label(root, text="")
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
