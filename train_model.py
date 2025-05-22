import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
config = {
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'backbones': ["EfficientNetB0", "MobileNetV2"]
}

# Corrected Paths (based on your structure)
splits_data_path = r'D:\ppp\Data\splits'
model_save_path = r'D:\ppp\Models'
plots_path = r'D:\ppp\Plots'
comparison_csv = r'D:\ppp\Results\model_comparison.csv'

# Classes
classes = ['Building', 'Field', 'Lawn', 'Mountain', 'Road', 'Vehicles', 'WaterArea', 'Wilderness']
num_classes = len(classes)

# Load data
def load_data():
    try:
        X_train = np.load(os.path.join(splits_data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(splits_data_path, 'y_train.npy'))
        X_val = np.load(os.path.join(splits_data_path, 'X_val.npy'))
        y_val = np.load(os.path.join(splits_data_path, 'y_val.npy'))
        X_test = np.load(os.path.join(splits_data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(splits_data_path, 'y_test.npy'))
        print("Data loaded successfully.")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Define models
def create_model(backbone_name, input_shape=(224, 224, 3)):
    if backbone_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported backbone name!")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=x)

# Compile and train model
def train_model(model, backbone_name, X_train, y_train, X_val, y_val, optimizer, save_path):
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[checkpoint, early_stopping]
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    plot_training(history, backbone_name, optimizer.__class__.__name__, y_train, model, X_train)
    return history, training_time

# Plot training
def plot_training(history, backbone_name, optimizer_name, y_train, model, X_train):
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{backbone_name} Accuracy ({optimizer_name})")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{backbone_name} Loss ({optimizer_name})")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, f"{backbone_name}_{optimizer_name}_training_plot.png"))
    plt.show()

    # Classification report
    report = classification_report(y_train, np.argmax(model.predict(X_train), axis=1), target_names=classes, output_dict=True)

    # Class-wise Precision
    plt.figure(figsize=(12, 6))
    plt.bar(classes, [report[cls]['precision'] for cls in classes], color='skyblue')
    plt.title(f"{backbone_name} Class-wise Precision ({optimizer_name})")
    plt.xlabel('Classes')
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(os.path.join(plots_path, f"{backbone_name}_{optimizer_name}_class_precision.png"))
    plt.show()

    # Class-wise Recall
    plt.figure(figsize=(12, 6))
    plt.bar(classes, [report[cls]['recall'] for cls in classes], color='lightgreen')
    plt.title(f"{backbone_name} Class-wise Recall ({optimizer_name})")
    plt.xlabel('Classes')
    plt.ylabel('Recall')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(os.path.join(plots_path, f"{backbone_name}_{optimizer_name}_class_recall.png"))
    plt.show()

# Evaluate model
def evaluate_model(model, X_test, y_test, backbone_name):
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{backbone_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, f"{backbone_name}_confusion_matrix.png"))
    plt.show()

    return report

# Save results
def save_results_to_csv(backbone_name, optimizer_name, report, training_time):
    results = {
        "Backbone": backbone_name,
        "Optimizer": optimizer_name,
        "Accuracy": report['accuracy'],
        "Recall": np.mean([report[cls]['recall'] for cls in classes]),
        "Precision": np.mean([report[cls]['precision'] for cls in classes]),
        "F1 Score": np.mean([report[cls]['f1-score'] for cls in classes]),
        "Training Time (s)": training_time
    }

    os.makedirs(os.path.dirname(comparison_csv), exist_ok=True)
    results_df = pd.DataFrame([results])

    if os.path.exists(comparison_csv):
        existing_df = pd.read_csv(comparison_csv)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    results_df.to_csv(comparison_csv, index=False)

# Main
def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    optimizers = [
        RMSprop(learning_rate=config['learning_rate']),
        Adam(learning_rate=config['learning_rate']),
        SGD(learning_rate=config['learning_rate'])
    ]

    for backbone in config['backbones']:
        for optimizer in optimizers:
            tf.keras.backend.clear_session()

            model = create_model(backbone)
            optimizer_name = optimizer.__class__.__name__
            save_path = os.path.join(model_save_path, f"{backbone}_{optimizer_name}_best.keras")

            print(f"\n--- Training {backbone} with {optimizer_name} ---")
            history, training_time = train_model(model, backbone, X_train, y_train, X_val, y_val, optimizer, save_path)

            print(f"Evaluating {backbone} with {optimizer_name}...")
            report = evaluate_model(model, X_test, y_test, backbone)

            print(f"Saving results for {backbone} with {optimizer_name}...")
            save_results_to_csv(backbone, optimizer_name, report, training_time)

if __name__ == "__main__":
    main()
