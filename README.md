# Weather Image classsification



A comprehensive deep learning framework for satellite image classification that implements and compares multiple state-of-the-art neural network architectures. The system provides automated training, evaluation, and comparison of Custom CNN, VGG19 Transfer Learning, ConvNeXt-inspired, and Swin Transformer-like models.

---

## Overview

This system addresses the challenge of automated satellite image classification by providing a unified framework that compares multiple deep learning architectures. It handles the complete machine learning pipeline from data preprocessing to model evaluation, making it suitable for both research and practical applications.

---

## Key Features

- **Multi-Architecture Comparison**: Simultaneous training and evaluation of four different deep learning models
- **Transfer Learning Implementation**: Pre-trained VGG19 with domain-specific fine-tuning
- **Modern Architecture Support**: ConvNeXt and Swin Transformer inspired implementations
- **Comprehensive Evaluation**: Complete performance analysis with multiple metrics
- **Automated Data Handling**: Built-in preprocessing, augmentation, and class imbalance correction
- **Visualization Suite**: Training curves, confusion matrices, and prediction analysis
- **Production Ready**: Model serialization and deployment preparation

---

## Tech Stack

| Component      | Technology / Tool                   |
|----------------|-------------------------------------|
| Model Training | TensorFlow, Keras                   |
| Notebook Dev   | Google Colab, Jupyter Notebooks     |
| Data Handling  | NumPy, Pandas, OpenCV               |
| Visualization  | Matplotlib, Seaborn                 |
| Evaluation     | scikit-learn                        |

---

## Supported Classifications

- **Cloudy Regions**: Cloud-covered geographical areas  
- **Desert Terrain**: Arid and semi-arid landscapes  
- **Green Areas**: Vegetation, forests, and agricultural land  
- **Water Bodies**: Rivers, lakes, and coastal areas  

---

## Model Architectures

### 1. Custom Convolutional Neural Network
Purpose-built CNN with three convolutional blocks, batch normalization, global average pooling, and dropout regularization.

### 2. VGG19 Transfer Learning
Pre-trained on ImageNet with a custom classification head and domain-specific fine-tuning.

### 3. ConvNeXt-Inspired Architecture
Modern CNN using depthwise convolutions, GELU activation, and efficient parameter design.

### 4. Swin Transformer-Like Model
Patch-based Transformer featuring hierarchical feature learning with MLP and self-attention blocks.

---

## Dataset Requirements

- Directory-structured multi-class image dataset  
- Supports formats: JPEG, PNG, BMP, TIFF  
- Automatic stratified split: 70% training, 15% validation, 15% testing  

---

## Performance Metrics

- **Accuracy**: Overall classification score  
- **Precision**: Positive predictive value  
- **Recall**: Sensitivity per class  
- **F1-Score**: Harmonic mean of precision and recall  
- **Specificity**: True negative rate  
- **Confusion Matrix**: Detailed class breakdown  

---

## Technical Implementation

### Data Preprocessing

- Image resizing and normalization  
- Data augmentation (rotation, shift, flip)  
- Class weight computation  
- Stratified splitting

### Training Pipeline

- Adam optimizer with learning rate scheduling  
- Early stopping and monitoring  
- Dropout and batch normalization  

### Evaluation Framework

- Cross-validation support  
- Statistical comparison of models  
- Visual performance plots and ranking

---

## System Requirements

### Software

- TensorFlow 2.x  
- NumPy, Pandas, scikit-learn  
- OpenCV  
- Matplotlib, Seaborn  

### Hardware

- GPU (CUDA-enabled recommended)  
- 8GB RAM minimum  
- SSD storage for fast image loading  

---

## Installation & Setup

Compatible with local and cloud environments (e.g., Google Colab). Includes ready-to-run notebooks and modular scripts.

> **Note:** The VGG19_Transfer_model.h5 was too large for GitHub. It can be regenerated using the training scripts.

---

## Applications

- Remote Sensing & Environmental Monitoring  
- Land Use and Urban Planning  
- Agriculture and Crop Assessment  
- Climate Impact Studies  
- Satellite Data Analysis for Research

---

## License

This project is licensed under the MIT License.

---

