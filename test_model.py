import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Load the trained model
MODEL_PATH = r"D:\ppp\final_best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (must match training dataset order)
class_labels = ["Building", "Field", "Mountain", "Road", "Vehicles", "WaterArea", "Wilderness"]

# Define test data directory
test_data_dir = r"D:\\ppp\\Data\\test"

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data preprocessing for test set (consistent with training preprocessing)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Assuming multi-class classification
    shuffle=False  # No shuffling for test data
)

# Verify class indices and print class distribution
class_indices = test_generator.class_indices
print("Class Indices:", class_indices)

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    image = image.resize(target_size)  # Resize
    print(f"Image size after resizing: {image.size}")  # Print image size for verification
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Get image path from user input
image_path = input("Enter the path to the test image: ")

# Preprocess and predict
processed_image = preprocess_image(image_path)
predictions = model.predict(processed_image)

# Get the predicted class and confidence
predicted_class = np.argmax(predictions)
confidence = float(np.max(predictions))

# Print the result
print(f"Predicted Class: {class_labels[predicted_class]}")
print(f"Confidence: {confidence:.4f}")
