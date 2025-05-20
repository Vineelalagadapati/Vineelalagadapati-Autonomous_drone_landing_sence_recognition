import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the model
model_path = "D:/Project/model.keras"
model = tf.keras.models.load_model(model_path)

# Load test images (Modify this path based on your test dataset)
test_dir = "D:/Project/Data/validation/"  # Change if needed

# Get class labels (assuming same folder names as training data)
class_labels = sorted(os.listdir(test_dir))  # Extract class names from folder structure

# Function to load and preprocess an image
def preprocess_image(img_path, target_size=(224, 224)):  # Update size if different
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Test on sample images
sample_image_path = os.path.join(test_dir, class_labels[0], os.listdir(os.path.join(test_dir, class_labels[0]))[0])
test_image = preprocess_image(sample_image_path)

# Get prediction
prediction = model.predict(test_image)
predicted_label = class_labels[np.argmax(prediction)]

# Print results
print(f"Actual Class: {class_labels[0]}")
print(f"Predicted Class: {predicted_label}")

# Evaluate model on full test set
test_loss, test_accuracy = model.evaluate(test_image, np.array([[1 if i == np.argmax(prediction) else 0 for i in range(len(class_labels))]]))

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
