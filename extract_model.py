import tensorflow as tf
import json

# Load the model
model = tf.keras.models.load_model("D:/Project/model.keras")

# Get model architecture as JSON
model_json = model.to_json()

# Save to a file
with open("D:/Project/model_architecture.json", "w") as json_file:
    json.dump(json.loads(model_json), json_file, indent=4)

print("Model architecture saved to model_architecture.json")
