# Real-Time Emotion Recognition with CNN

This project implements a deep learning system that can detect emotions from face images in real-time using a webcam. We trained a CNN from scratch on the FER-2013 dataset and built a live demo that predicts emotions as you look into the camera.

## What This Does

The model classifies emotions into 7 categories: angry, disgust, fear, happy, neutral, sad, and surprise. You can train it yourself and then run a live webcam demo that detects faces and shows the predicted emotion in real-time.

## Dataset

We're using the [FER-2013 dataset from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). It has about 35k grayscale images of faces (48×48 pixels) labeled with emotions.

## Project Structure

```
emotion_recognition_project/
├── data/                 # FER-2013 dataset (train/val/test)
├── models/               # Trained models saved here
├── notebooks/
│   ├── run_training.ipynb    # Run training on Colab
│   └── run_inference.ipynb   # Live webcam demo
├── scripts/
│   └── split_validation.py   # Preprocessing script
└── src/
    ├── train.py          # Training pipeline
    ├── main.py           # Real-time webcam demo
    ├── model.py          # CNN architecture
    ├── utils.py          # Helper functions (preprocessing, face detection)
    └── config.py         # Hyperparameters
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

You'll need PyTorch, OpenCV, numpy, and some other libraries. If you're on Google Colab, most of these come pre-installed.

## Experiments

We tested different configurations:
- **Optimizers**: Adam vs SGD
- **Regularization**: Effect of Dropout
- **Normalization**: Effect of BatchNorm


## Notes

- The model works with grayscale 48×48 face images
- For the live demo, we use OpenCV's face detector (Haar cascade) to find faces in the webcam feed

