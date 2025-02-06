from ultralytics import YOLO

from pathlib import Path

# Define paths
dataset_path = "/Users/mrlee/development/garbage_classification/garbage_classification/data/raw"

for split in ["train", "val", "test"]:
    images = list(Path(f"{dataset_path}/{split}/images").glob("*.jpg"))
    labels = list(Path(f"{dataset_path}/{split}/labels").glob("*.txt"))

    print(f"\nChecking {split} set:")
    print(f" - Found {len(images)} images")
    print(f" - Found {len(labels)} labels")

    # Check for missing labels
    missing_labels = [img.stem for img in images if not Path(f"{dataset_path}/{split}/labels/{img.stem}.txt").exists()]
    if missing_labels:
        print(f"❌ Missing labels for {len(missing_labels)} images!")
        print(missing_labels[:10])  # Show first 10 missing labels
    else:
        print("✅ All images have corresponding labels.")

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model as the base

# Define dataset path
DATASET_YAML = "dataset.yaml"

# Train the model
print("Starting YOLOv8 training...")
# results = model.train(data=DATASET_YAML, epochs=50, imgsz=640, batch=16)
results = model.train(data=DATASET_YAML, epochs=10, imgsz=320, batch=8, verbose=True, device="mps") # For mac


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