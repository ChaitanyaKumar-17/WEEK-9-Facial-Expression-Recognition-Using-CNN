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