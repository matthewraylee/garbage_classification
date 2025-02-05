import torch
import torchvision
from torchvision import transforms
import gradio as gr

# Load pre-trained ResNet model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# ImageNet class labels
import requests
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.split("\n")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image):
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Create results dictionary
    results = {labels[idx]: float(prob) for prob, idx in zip(top5_prob, top5_catid)}

    return results

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classification with ResNet50",
    description="Upload an image to see what the model thinks it contains!",
    examples=[
        ["example1.jpg"],
        ["example2.jpg"]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()