import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],  # You can also add 'poly' here
}

# Create SVM model
svm = SVC()

# Grid search for best parameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the SVM model: {accuracy:.2f}')

# Function to preprocess the input image and make predictions
def predict_digit(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((8, 8))  # Resize to 8x8 for the model
    img = np.array(img, dtype=np.float32)
    img = img.flatten().reshape(1, -1)  # Flatten the image

    img_scaled = scaler.transform(img)  # Scale the image
    prediction = best_model.predict(img_scaled)
    return prediction[0]

# Create the GUI
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_digit = predict_digit(file_path)
        result_label.config(text=f"The predicted digit is: {predicted_digit}")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognizer with SVM")
root.geometry("400x200")

upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

result_label = Label(root, text="")
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
