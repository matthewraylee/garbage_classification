from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano

# Define dataset path
DATASET_YAML = "YOLO-Waste-Detection-1/data.yaml"

# Train the model
print("Starting YOLOv8 training...")
# results = model.train(data=DATASET_YAML, epochs=50, imgsz=640, batch=16, device="mps")  # Use MPS for Apple Silicon
results = model.train(
    data=DATASET_YAML,  
    epochs=10,  # Reduced for speed
    imgsz=320,  # Lower resolution for faster training
    batch=8,  # Avoids MPS memory issues
    device="mps",  # Runs on Apple M2 Pro GPU
    amp=True,  # Enables mixed precision for speed
    workers=2,  # Prevents excessive CPU usage
    augment=False,  # Disables unnecessary data augmentation
    verbose=False  # Reduces console clutter
)


# Print training results
print("\nTraining Complete!")
print(f"Final Training Loss: {results.results_dict.get('metrics/loss', 'N/A')}")
print(f"Best Validation mAP50-95: {results.results_dict.get('metrics/mAP_50-95(B)', 'N/A'):.4f}")
print(f"Best Validation mAP50: {results.results_dict.get('metrics/mAP_50(B)', 'N/A'):.4f}")

# Validate the model
print("\nRunning validation...")
val_results = model.val(data=DATASET_YAML)

# Print validation metrics
print("\nValidation Complete!")
print(f"Validation mAP50-95: {val_results.results_dict.get('metrics/mAP_50-95', 'N/A'):.4f}")
print(f"Validation mAP50: {val_results.results_dict.get('metrics/mAP_50', 'N/A'):.4f}")

# Run inference on the test set
print("\nRunning test evaluation...")
test_results = model.val(data=DATASET_YAML, split='test')

# Print test results
print("\nTest Evaluation Complete!")
print(f"Test mAP50-95: {test_results.results_dict.get('metrics/mAP_50-95', 'N/A'):.4f}")
print(f"Test mAP50: {test_results.results_dict.get('metrics/mAP_50', 'N/A'):.4f}")
print(f"Test Precision: {test_results.results_dict.get('metrics/precision', 'N/A'):.4f}")
print(f"Test Recall: {test_results.results_dict.get('metrics/recall', 'N/A'):.4f}")