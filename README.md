# Sign Language Interpreter

## Overview
The Sign Language Interpreter is a real-time web application that detects and translates hand signs into text using computer vision and machine learning. The system captures hand landmarks from a webcam, processes them using a trained classification model, and displays predictions through an interactive user interface.

---

## Features
- Real-time hand sign detection using webcam
- Machine learning model trained on custom dataset
- Supports sign recognition for letters: A, B, C, D, E
- Live prediction overlay with continuous updates
- Sentence builder to construct words from detected signs
- Interactive UI with controls (Add, Space, Delete, Clear)
- Flask-based backend for model inference

---

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript (Webcam + API communication)

### Backend
- Python
- Flask

### Machine Learning & Computer Vision
- OpenCV
- MediaPipe (Hand Landmark Detection)
- Scikit-learn (Model Training)
- NumPy
- Pandas

---

## Project Structure
