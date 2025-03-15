---
title: Waste Classification Demo
emoji: 🗑️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
---

# Waste Classification Demo

This application uses YOLOv8 to classify waste items into three categories:

- **Compost**: Biodegradable items (green)
- **Recyclable**: Items that can be processed and reused (blue)
- **Garbage**: Items that cannot be composted or recycled (red)

## How to Use

1. Allow camera access when prompted
2. Point your camera at waste items
3. View real-time classification results with explanations

## Development Information

### Setup

poetry install

### Data Import

kaggle datasets download asdasdasasdas/garbage-classification -p garbage_classification/data/ --unzip

### Model Training

yolo task=detect mode=train model=yolov8n.pt data=YOLO-Waste-Detection-2/data.yaml epochs=50 batch=16 imgsz=640 patience=5

### Model Testing

yolo task=detect mode=val model=runs/detect/train24/weights/best.pt data=datasets/data.yaml
