# Sign Language Translator — Deep Learning

> **Project:** Real-time sign language translator that recognizes hand signs (alphabet/words/phrases) using deep learning and optionally converts recognized signs to speech.

---

## Table of contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset Options](#dataset-options)
4. [Model Architecture (Suggested)](#model-architecture-suggested)
5. [Folder structure](#folder-structure)
6. [Requirements](#requirements)
7. [Setup & Installation](#setup--installation)
8. [Preparing the dataset](#preparing-the-dataset)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Real-time Inference / Demo](#real-time-inference--demo)
12. [Tips to Improve Accuracy](#tips-to-improve-accuracy)
13. [Troubleshooting](#troubleshooting)
14. [Citation & License](#citation--license)
15. [Contact](#contact)

---

## Project Overview

This repository demonstrates how to build a sign language translator using computer vision and deep learning. The pipeline typically includes:

* Data collection and preprocessing (images / video frames or landmark keypoints)
* A model to classify signs (CNN, CNN+LSTM, or Transformer on extracted keypoints)
* Optional post-processing to form words/phrases
* Optional Text-to-Speech to speak the recognized text

The code is designed to work in both offline (train/evaluate) and real-time (webcam) modes.

---

## Features

* Trainable sign recognition model (alphabet or custom gestures)
* Support for image-based or keypoint-based inputs (e.g., Mediapipe hand landmarks)
* Real-time webcam inference with bounding-box and label overlay
* Optional Text-to-Speech (TTS) output for translations
* Configurable hyperparameters and model checkpoints

---

## Dataset Options

You can use any of the following datasets or collect your own dataset:

* **Sign Language MNIST** (handwritten-like images for ASL alphabet)
* **ASL Alphabet Dataset** (image dataset with 26 letters + other signs)
* **Custom dataset**: capture video frames from webcam with multiple labeled folders (one folder per class)

> Tip: For real-time performance it's common to use Mediapipe to extract 21 hand landmarks per hand and train a small model on the landmark coordinates rather than raw images.

---

## Model Architecture (Suggested)

Here are two suggested approaches:

### 1) Image-based (CNN)

* Input: 224x224 RGB image (cropped hand region)
* Backbone: MobileNetV2 / EfficientNet-lite or simple ConvNet
* Head: GlobalAveragePooling -> Dense(128, ReLU) -> Dropout -> Dense(num_classes, softmax)

### 2) Keypoint-based (Lightweight and real-time)

* Input: Flattened hand landmark vector (e.g., 21 keypoints × 2 or 3 = 42 or 63 dims)
* Model: Fully-connected MLP or a small temporal model (1D Conv / LSTM) for sequences
* Example: [Input] -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(num_classes, softmax)

**Why keypoints?** Smaller models, faster inference, more invariant to background/lighting.

---

## Folder structure

```
sign-language-translator/
├── data/                  # raw + processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/             # Jupyter notebooks (EDA, experiments)
├── models/                # saved model checkpoints
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py       # real-time webcam demo
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements

A minimal `requirements.txt` (example):

```
opencv-python
mediapipe
tensorflow>=2.6
numpy
pandas
matplotlib
scikit-learn
gTTS                 # optional, for text-to-speech
sounddevice          # optional, for TTS playback
```

Use a virtual environment (venv / conda) for reproducibility.

---

## Setup & Installation

```bash
# create venv (optional)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate         # Windows

# install dependencies
pip install -r requirements.txt
```

---

## Preparing the dataset

### Image-based dataset

1. Organize images into class folders: `data/raw/A/`, `data/raw/B/`, ...
2. Use `src/preprocess.py` to resize/crop and augment images (horizontal flip, rotations, brightness jitter).

### Keypoint-based dataset (recommended if using Mediapipe)

1. Run a capture script that records frames from webcam and saves hand landmarks for each label.
2. Store per-sample JSON or CSV with normalized landmark coordinates.

Example landmark normalization:

* Translate so wrist is origin
* Scale by maximum distance between landmarks
* Optionally include handedness (left/right)

---

## Training

Example training command (image-based CNN):

```bash
python src/train.py \
  --data_dir data/processed/images \
  --model_dir models/cnn_alpha \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-4
```

Example training command (keypoint MLP):

```bash
python src/train.py \
  --data_dir data/processed/keypoints \
  --model_dir models/keypoint_mlp \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 1e-3
```

Suggested callbacks in `train.py`:

* ModelCheckpoint (save best)
* EarlyStopping (monitor val_loss)
* TensorBoard (for visualization)

Hyperparameters to tune:

* Learning rate, optimizer (Adam/SGD)
* Model size (number of layers/neurons)
* Sequence length if using temporal models

---

## Evaluation

Run evaluation script to compute accuracy, confusion matrix, precision/recall per class:

```bash
python src/evaluate.py --model models/keypoint_mlp/best.h5 --test_dir data/processed/keypoints/test
```

Plot a confusion matrix to identify commonly confused signs and gather more data for them.

---

## Real-time Inference / Demo

The `src/inference.py` should:

1. Read webcam frames (OpenCV)
2. Detect hand and extract ROI or landmarks (Mediapipe)
3. Preprocess input and feed to trained model
4. Display predicted label on frame
5. Optionally accumulate predictions into words and use gTTS to speak the result

Example run:

```bash
python src/inference.py --model models/keypoint_mlp/best.h5 --use_mediapipe
```

Notes for better UX:

* Apply temporal smoothing (majority vote across last N frames) to reduce flicker.
* Show confidence score and an indicator when the model is uncertain.

---

## Tips to Improve Accuracy

* Collect diverse data from multiple users, backgrounds, and lighting.
* Use data augmentation and class balancing.
* For phrase/word-level translation use sequence models (LSTM, GRU) or Transformers on a window of successive frames/landmarks.
* Use transfer learning (MobileNetV2/EfficientNet) for image-based models if dataset is small.
* Label-cleaning: remove ambiguous or low-quality frames.

---

## Troubleshooting

* **Low accuracy**: Check data quality, class imbalance, model capacity.
* **Webcam feed slow**: run a lighter model (keypoint-based MLP) or reduce input resolution.
* **Mediapipe failing to detect hands**: ensure proper lighting and that the hand is fully in frame.

---

## Citation & License

If you use third-party datasets or models, cite them. Consider licensing this repo with MIT License:

```
MIT License
Copyright (c) 2025 Your Name
```

---

## Contact

Created by: **Your Name**

If you want improvements, additional features (phrase translation, multi-language TTS, mobile app), or help training on your own data, open an issue or contact: [your.email@example.com](mailto:your.email@example.com)

---

### Useful commands summary

* Train: `python src/train.py --data_dir data/processed --model_dir models/ --epochs 50`
* Evaluate: `python src/evaluate.py --model models/best.h5 --test_dir data/processed/test`
* Run demo: `python src/inference.py --model models/best.h5 --use_mediapipe`

---

Good luck! If you'd like, I can also:

* generate `src/train.py` / `src/inference.py` starter scripts,
* create a `requirements.txt`,
* or convert this README into a nicely formatted project document.
