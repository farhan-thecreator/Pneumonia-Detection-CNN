# Pneumonia-Detection-CNN
A CNN model for classifying chest X-rays into Normal and Pneumonia categories.

# Pneumonia Detection using Chest X-rays (CNN)

This project focuses on using deep learning for detecting Pneumonia from chest X-ray images. It involves training a Convolutional Neural Network (CNN) on a labeled dataset to classify images as either “Pneumonia” or “Normal”.

---

## Dataset

- Source: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Total images: ~5,800
- Classes:
  - Normal
  - Pneumonia

*Due to the size (~2GB), the dataset is not included in the repository.*

---

## Model Overview

- CNN built from scratch using Keras & TensorFlow
- Layers used: Conv2D, MaxPooling, Flatten, Dropout, Dense
- Activation: ReLU and Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall, F1 Score

---

## Results

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Normal    | 0.37      | 0.18   | 0.24     |
| Pneumonia | 0.62      | 0.82   | 0.71     |

- **Overall Accuracy:** 58%
- **Confusion Matrix:**  
  ![Confusion Matrix](images/confusion_matrix.png)

> The model performs significantly better on detecting Pneumonia but struggles with "Normal" due to class imbalance.

---

## Visualizations

- Training/Validation Accuracy & Loss graphs
- Confusion Matrix
- Classification Report

---

## Key Learnings

- Class imbalance needs to be addressed (via augmentation or class weights)
- More training epochs or better architectures (like VGG/ResNet) could boost performance
- Transfer learning is a strong future direction

---

## How to Run

This project is tested in Google Colab.

```python
# Install dependencies
!pip install tensorflow keras matplotlib opencv-python seaborn
