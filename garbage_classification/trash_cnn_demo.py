# import torch
# import torchvision
# from torchvision import transforms
# import gradio as gr
# from ultralytics import YOLO

# # Load pre-trained ResNet model
# # model = torchvision.models.resnet50(pretrained=True)
# # model.eval()

# # ImageNet class labels
# import requests
# LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# labels = requests.get(LABELS_URL).text.split("\n")

# # Image preprocessing
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# def predict_image(image):
#     # Preprocess the image
#     image_tensor = preprocess(image).unsqueeze(0)

#     # Make prediction
#     with torch.no_grad():
#         output = model(image_tensor)

#     # Get probabilities
#     probabilities = torch.nn.functional.softmax(output[0], dim=0)

#     # Get top 5 predictions
#     top5_prob, top5_catid = torch.topk(probabilities, 5)

#     # Create results dictionary
#     results = {labels[idx]: float(prob) for prob, idx in zip(top5_prob, top5_catid)}

#     return results

# # Create Gradio interface
# iface = gr.Interface(
#     fn=predict_image,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(num_top_classes=5),
#     title="Image Classification with ResNet50",
#     description="Upload an image to see what the model thinks it contains!",
#     examples=[
#         ["example1.jpg"],
#         ["example2.jpg"]
#     ]
# )

# # Launch the app
# if __name__ == "__main__":
#     iface.launch()

####################################### Yolo v8 #######################################

import torch
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# Load trained YOLO model
model_dir = "runs/detect/train11/weights/best.pt"  # Change this to your model path
model = YOLO(model_dir)  # Load custom YOLO model

def predict_image(image):
    # Convert image to YOLO format (PIL image)
    results = model(image)

    # Extract detections
    detections = results[0].boxes  # Bounding boxes, confidence scores, and class indices
    class_names = results[0].names  # Class names from the model

    predictions = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        label = class_names[class_id]  # Class label

        predictions.append({
            "label": label,
            "confidence": conf,
            "bounding_box": [x1.item(), y1.item(), x2.item(), y2.item()]
        })

    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Object Detection with YOLO",
    description="Upload an image and the model will detect objects in it.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()