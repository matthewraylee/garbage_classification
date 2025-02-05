from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model as the base

# Define dataset path
DATASET_YAML = "dataset.yaml"

# Train the model
print("🚀 Training YOLOv8 Model...")
results = model.train(data=DATASET_YAML, epochs=50, imgsz=640, batch=16)

# Print training results
print("\n✅ Training Complete!")
print(f"📉 Final Training Loss: {results.results_dict['metrics/loss']}")
print(f"📊 Best Validation mAP50-95: {results.results_dict['metrics/mAP_50-95(B)']:.4f}")
print(f"📊 Best Validation mAP50: {results.results_dict['metrics/mAP_50(B)']:.4f}")

# Validate the model
print("\n🔍 Running Validation...")
val_results = model.val(data=DATASET_YAML)

# Print validation metrics
print("\n✅ Validation Complete!")
print(f"🎯 Validation mAP50-95: {val_results.results_dict['metrics/mAP_50-95']:.4f}")
print(f"🎯 Validation mAP50: {val_results.results_dict['metrics/mAP_50']:.4f}")

# Run inference on the test set
print("\n🧪 Running Test Evaluation...")
test_results = model.val(data=DATASET_YAML, split='test')

# Print test results
print("\n✅ Test Evaluation Complete!")
print(f"📝 Test mAP50-95: {test_results.results_dict['metrics/mAP_50-95']:.4f}")
print(f"📝 Test mAP50: {test_results.results_dict['metrics/mAP_50']:.4f}")
print(f"🔍 Test Precision: {test_results.results_dict['metrics/precision']:.4f}")
print(f"🎯 Test Recall: {test_results.results_dict['metrics/recall']:.4f}")