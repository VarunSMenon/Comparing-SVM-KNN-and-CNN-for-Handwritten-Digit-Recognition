import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Label, Button
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image, ImageFilter
import os

# Load the dataset and normalize the data
digits = load_digits()
X, y = digits.data, digits.target

# Standardize the dataset (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use GridSearchCV to tune K value and improve accuracy
param_grid = {'n_neighbors': np.arange(1, 25)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)  # 5-fold cross-validation
knn_cv.fit(X_train, y_train)

# Predict on test data
y_pred = knn_cv.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Best K value: {knn_cv.best_params_['n_neighbors']}")
print(f"Test accuracy with best K: {accuracy * 100:.2f}%")

# Function to predict on custom image
def predict_digit(image_path):
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((8, 8), Image.Resampling.LANCZOS)  # Resize to 8x8 like the digits dataset
        img = img.filter(ImageFilter.GaussianBlur(1))  # Optional: Apply a slight blur for smoothing
        img = np.array(img) / 16.0  # Scale pixel values to match the dataset's range (0-16)

        # Normalize the input image
        img_rescaled = scaler.transform([img.flatten()])

        # Predict the digit
        prediction = knn_cv.predict(img_rescaled)
        return prediction[0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Create the GUI
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_digit = predict_digit(file_path)
        if predicted_digit is not None:
            result_label.config(text=f"The predicted digit is: {predicted_digit}")
        else:
            result_label.config(text="Failed to predict. Please try again with a different image.")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("400x200")

upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

result_label = Label(root, text="")
result_label.pack(pady=20)

root.mainloop()
