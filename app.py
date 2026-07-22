import os
import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import datetime

# Step 1 Data Preparation & Preprocessing

print("Downloading FER-2013 dataset via Kagglehub...")
# This automatically downloads the dataset and caches it locally
dataset_path = kagglehub.dataset_download("msambare/fer2013") 
print(f"Dataset downloaded to: {dataset_path}")

# The downloaded directory automatically contains 'train' and 'test' subdirectories
TRAIN_DATA_DIR = f"{dataset_path}/train"
VAL_DATA_DIR = f"{dataset_path}/test"

# Load directly into TensorFlow, keeping the RGB (3-channel) requirement
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(48, 48),
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DATA_DIR,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(48, 48),
    batch_size=32
)

class_names = train_dataset.class_names
print(f"Emotion Classes: {class_names}")

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# STEP 2: Model Architecture

print("Building the CNN model...")

model = models.Sequential([
    layers.InputLayer(input_shape=(48, 48, 3)),
    
    # Data Augmentation (Only active during training)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    # Block 1
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten -> Dense -> Dropout
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output Layer
    layers.Dense(7, activation='softmax')
])

model.summary()

# STEP 3: Compilation & Training

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup Callbacks for better training management
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping prevents overfitting if validation loss stops decreasing
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

# Reduce learning rate when the model gets stuck
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6
)

print("Starting training for up to 50 epochs (Early Stopping enabled)...")
# Train the model with the new callbacks
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50, # Increased epochs because early stopping will catch it
    callbacks=[tensorboard_callback, early_stopping, reduce_lr]
)

# STEP 4: Evaluation & Visualization

# 4A: Plot training vs validation accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 4B: Display Confusion Matrix
print("Generating Confusion Matrix...")
y_true = []
y_pred_probs = []

for images, labels in val_dataset:
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Emotion')
plt.xlabel('Predicted Emotion')
plt.show()

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# 4C: Visualize Filters (Feature Maps)
print("Visualizing feature maps for the first Convolutional Layer...")

# Extract the outputs of the first Conv2D layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs[0])

for images, labels in val_dataset.take(1):
    sample_image = images[0:1] 
    break

activations = activation_model.predict(sample_image)

plt.figure(figsize=(10, 10))
for i in range(16): 
    plt.subplot(4, 4, i + 1)
    plt.imshow(activations[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('First Conv2D Layer Feature Maps', fontsize=16)
plt.show()