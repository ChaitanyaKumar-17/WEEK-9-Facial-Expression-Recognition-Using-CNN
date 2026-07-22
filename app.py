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

