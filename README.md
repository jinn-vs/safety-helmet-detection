# Helmet Detection System

Safety Helmet Detection System using YOLOv8

## Overview
This project detects whether a person is wearing a safety helmet or not using a two-model pipeline approach:
- COCO YOLOv8n for person detection
- Custom YOLOv8n for helmet and no-helmet detection

The system identifies persons in an image, then checks whether each person is wearing a helmet.
Persons without helmets are marked as UNSAFE.

## Results
| Metric | Value |
|--------|-------|
| Helmet mAP@0.5 | 97.7% |
| No-helmet mAP@0.5 | 92.6% |

## Dataset
- Total Images: 4,600+
- Sources: Hard Hat Universe, IIT Goa Helmet Dataset, Kaggle Bikes Helmets Dataset
- Classes: helmet, no-helmet, person

## Tech Stack
Python, YOLOv8 (Ultralytics), Google Colab, Gradio, OpenCV

## How to Run
    pip install ultralytics gradio opencv-python
    python app.py

## Developed By
Muhammad Ali Qamer
AIRI Team PITB - AI/ML Internship

## License
MIT License