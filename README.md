# 🎯 Facial Expression Recognition using CNN

A robust Python machine learning project designed to ingest facial image data, preprocess and augment features, and build a convolutional neural network (CNN) model to classify human emotions. This project emphasizes industry best practices in computer vision, deep learning architectures, and model evaluation.

---

## 💡 Overview

Modern applications rely on computer vision to identify distinct facial expressions for sentiment analysis, human-computer interaction, and accessibility tools. This project automates a workflow for extracting visual features from image datasets. By applying data augmentation, utilizing deep convolutional layers, and visualizing the network's feature maps, the model transforms raw pixel data into classified emotion profiles.

---

## ✨ Key Features

*   **Automated Data Acquisition:** Utilizes the `kagglehub` API to seamlessly download and extract the FER-2013 dataset directly into the local environment.
*   **Data Augmentation:** Implements built-in Keras augmentation layers to dynamically transform images during training, specifically utilizing:
    *   `RandomFlip("horizontal")`
    *   `RandomRotation(0.1)`
    *   `RandomZoom(0.1)`
*   **Deep CNN Architecture:** Constructs a multi-block convolutional network utilizing `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers to extract hierarchical features.
*   **Advanced Training Callbacks:** Integrates `EarlyStopping` to monitor validation loss, `ReduceLROnPlateau` to dynamically adjust the learning rate, and `TensorBoard` for real-time training visualization.
*   **Comprehensive Visualization:** Outputs confusion matrices, classification reports, and visualizations of the first convolutional layer's feature maps to interpret the model's behavior.

---

## 🛠️ Prerequisites

Ensure you have the following installed before running the project:
*   Python 3.8 or higher
*   A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
*   Core Scientific and Deep Learning Libraries: `tensorflow`, `numpy`, `pandas`, `kagglehub`, `scikit-learn`, `seaborn`, `matplotlib`

---

## 🚀 Quick Start Guide

1.  Clone this repository to your local machine.
2.  Open your terminal or command prompt and install the necessary dependencies:
    ```bash
    pip install tensorflow numpy pandas kagglehub scikit-learn seaborn matplotlib
    ```
3.  Launch your Python environment or execute the script directly:
    ```bash
    python app.py
    ```

---

## 📊 Expected Output

Upon successful execution, the script will process the image data in memory and output visual insights, including:

*   **Training Diagnostics:** Visual outputs displaying the accuracy and loss curves over the training epochs.
*   **Model Evaluation Visuals:** A detailed Confusion Matrix heatmap and a printed Classification Report mapping predicted emotions against actual labels.
*   **Feature Map Extraction:** A 4x4 grid displaying 16 isolated feature maps from the network's first Convolutional layer, demonstrating how the model applies visual filters to the input image.

---

## 🧩 Pipeline Architecture: How It Works

This script serves as a practical application of standard deep learning pipelines tailored for computer vision tasks:

1.  **Data Ingestion & Preprocessing:** The script loads the FER-2013 dataset and scales the continuous pixel values (0-255) using a `Rescaling(1./255)` layer. The images are processed in batches of 32 with a fixed input shape of (48, 48, 3).
2.  **In-Model Augmentation:** The data passes through `RandomFlip`, `RandomRotation`, and `RandomZoom` layers applied exclusively during the training phase.
3.  **Convolutional Block 1:** 
    *   `Conv2D` (64 filters, 3x3)
    *   `BatchNormalization`
    *   `Conv2D` (64 filters, 3x3)
    *   `BatchNormalization`
    *   `MaxPooling2D` (2x2)
    *   `Dropout` (0.25)
4.  **Convolutional Block 2:** 
    *   `Conv2D` (128 filters, 3x3)
    *   `BatchNormalization`
    *   `Conv2D` (128 filters, 3x3)
    *   `BatchNormalization`
    *   `MaxPooling2D` (2x2)
    *   `Dropout` (0.25)
5.  **Convolutional Block 3:** 
    *   `Conv2D` (256 filters, 3x3)
    *   `BatchNormalization`
    *   `MaxPooling2D` (2x2)
    *   `Dropout` (0.25)
6.  **Dense Classification Head:** The spatial data is processed by a `Flatten` layer, passed through a `Dense` (256) layer with `BatchNormalization` and `Dropout` (0.5), and finalized in a `Dense` (7) Output Layer using the Softmax activation function for multi-class categorization.