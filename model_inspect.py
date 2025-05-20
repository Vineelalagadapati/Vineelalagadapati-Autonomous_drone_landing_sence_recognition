import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("D:/Project/model.keras")

# Print model summary
model.summary()
# Get model architecture
model_json = model.to_json()
print(model_json)
