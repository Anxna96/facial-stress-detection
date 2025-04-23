# Stress Detection with CNN and Real-Time Face Analysis

This project is about implementing a stress detection system based on facial expressions through Convolutional Neural Networks (CNN) and real-time analysis using MediaPipe.

## Project Structure

project/  
├── stress_detection.py  
├── stress_detection_model.h5  
├── requirements.txt  
├── README.md  
└── FER2013CK_filtered/  
  ├── train/  
  └── test/

##  Description of the project

The aim of this project is to build a binary classification model to distinguish and classify facial emotions into two categories: Stress and No Stress, using images from the FER2013 and CK+ databases, which have been compiled into a single folder. The code was also adapted to detect stress in faces in real time using a webcam.

Emotions are mapped as follows:

- **Stress:** angry, fear and disgust  
- **No Stress:** happy, neutral, surprise and sad

# Techniques 

- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Image processing with OpenCV
- Face detection with MediaPipe
- Visualization with Matplotlib & Seaborn

## Model used

- Model: Simple CNN with 3 convolutional layers and 2 dense layers  
- Image size: 48x48 pixels  
- Final activation: sigmoid (binary classification)  
- Loss function: binary_crossentropy  
- Optimizer: adam

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
