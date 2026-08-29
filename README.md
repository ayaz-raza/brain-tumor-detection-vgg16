# Brain Tumor Detection from MRI Images using VGG16

A deep learning project for detecting brain tumors from MRI images using
VGG16 transfer learning, fine-tuning, and Grad-CAM explainability.

## Overview

This project develops a binary image classification model to distinguish
between:

- **Tumor**
- **No Tumor**

The original dataset contains four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The three tumor categories are combined into a single **Tumor** class,
creating a binary classification problem.

## Methodology

The project follows these steps:

1. Dataset loading and exploration
2. Class distribution analysis
3. Conversion from four classes to binary classification
4. Image resizing to 224 × 224
5. Data augmentation
6. VGG16 transfer learning using ImageNet weights
7. Training with class weights
8. Fine-tuning of the final VGG16 convolutional block
9. Model evaluation
10. ROC-AUC analysis
11. Grad-CAM visualization for model interpretability

## Model Architecture

```text
Input Image (224 × 224 × 3)
            ↓
         VGG16
            ↓
Global Average Pooling
            ↓
       Dense (256)
            ↓
       Dropout (0.5)
            ↓
    Sigmoid Output
            ↓
    Tumor / No Tumor
