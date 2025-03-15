from ultralytics import YOLO
import yaml
import os
# Paths to dataset and hyperparameters
DATASET_YAML = os.path.join("YOLO-Waste-Detection-2", "dataset.yaml")  # Relative path to dataset.yaml
HYP_YAML = os.path.join("hyp.yaml")       # Relative path to hyp.yaml
# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
# Load hyperparameters from hyp.yaml
with open(HYP_YAML, 'r') as f:
    hyp = yaml.safe_load(f)  # Load as a dictionary
# Start training
print("Starting YOLOv8 training with custom hyperparameters...")
results = model.train(
    data=DATASET_YAML,        # Path to dataset config
    epochs=50,                # Number of epochs
    imgsz=640,                # Image size
    batch=16,                 # Batch size
    device=0,                 # Use GPU (0 for first GPU, -1 for CPU)
    amp=True,                 # Enable mixed precision training for faster performance
    workers=4,                # Number of workers for data loading
    optimizer="AdamW",        # Use AdamW optimizer
    lr0=hyp.get("lr0", 0.01),               # Initial learning rate
    lrf=hyp.get("lrf", 0.01),               # Final learning rate fraction
    momentum=hyp.get("momentum", 0.937),     # Momentum
    weight_decay=hyp.get("weight_decay", 0.0005), # Regularization
    hsv_h=hyp.get("hsv_h", 0.015),           # HSV-Hue augmentation
    hsv_s=hyp.get("hsv_s", 0.7),             # HSV-Saturation augmentation
    hsv_v=hyp.get("hsv_v", 0.4),             # HSV-Value augmentation
    translate=hyp.get("translate", 0.1),     # Translation
    scale=hyp.get("scale", 0.5),             # Scaling
    fliplr=hyp.get("fliplr", 0.5),           # Horizontal flip
    mosaic=hyp.get("mosaic", 1.0),           # Mosaic augmentation
    mixup=hyp.get("mixup", 0.1),             # Mixup augmentation
    augment=True,                            # Enable data augmentation
    verbose=True                             # Print detailed output
)
# Training Results
print("\nTraining Complete!")
print(f"Final Training Loss: {results.results_dict.get('metrics/loss', 'N/A')}")
print(f"Best Validation mAP50-95: {results.results_dict.get('metrics/mAP_50-95(B)', 'N/A'):.4f}")
print(f"Best Validation mAP50: {results.results_dict.get('metrics/mAP_50(B)', 'N/A'):.4f}")
# Run validation after training
print("\nRunning validation...")
val_results = model.val(data=DATASET_YAML)
# Validation Results
print("\nValidation Complete!")
print(f"Validation mAP50-95: {val_results.results_dict.get('metrics/mAP_50-95', 'N/A'):.4f}")
print(f"Validation mAP50: {val_results.results_dict.get('metrics/mAP_50', 'N/A'):.4f}")
# Run inference on test set
print("\nRunning test evaluation...")
test_results = model.val(data=DATASET_YAML, split='test')
# Test Results
print("\nTest Evaluation Complete!")
print(f"Test mAP50-95: {test_results.results_dict.get('metrics/mAP_50-95', 'N/A'):.4f}")
print(f"Test mAP50: {test_results.results_dict.get('metrics/mAP_50', 'N/A'):.4f}")
print(f"Test Precision: {test_results.results_dict.get('metrics/precision', 'N/A'):.4f}")
print(f"Test Recall: {test_results.results_dict.get('metrics/recall', 'N/A'):.4f}")
