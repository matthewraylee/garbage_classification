import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time

# Create a "styles" directory if it doesn't exist
os.makedirs("styles", exist_ok=True)

# Load trained YOLO model
model_dir = "runs/detect/train12/weights/best.pt"  # Change this to your model path
model = YOLO(model_dir)

# Define waste categories and their reasoning
waste_categories = {
    "compost": ["Organic", "Wood", "Paper", "Paper bag", "Paper cups", "Cellulose"],
    "recyclable": [
        "Cardboard", "Glass bottle", "Aluminum can", "Plastic bottle", "Plastic bag",
        "Plastic cup", "Plastic caps", "Scrap metal", "Paper", "Tetra pack", 
        "Aluminum caps", "Milk bottle"
    ],
    "garbage": [
        "Aerosols", "Ceramic", "Combined plastic", "Container for household chemicals",
        "Disposable tableware", "Electronics", "Foil", "Furniture", "Iron utensils",
        "Liquid", "Metal shavings", "Paper shavings", "Papier mache", "Plastic can",
        "Plastic canister", "Plastic shaker", "Plastic shavings", "Plastic toys",
        "Postal packaging", "Printing industry", "Stretch film", "Textile", "Tin",
        "Unknown plastic", "Zip plastic bag"
    ]
}

# Reasoning for each category
category_reasoning = {
    "compost": "This item is biodegradable and can be broken down naturally into compost.",
    "recyclable": "This item can be processed and reused to make new products.",
    "garbage": "This item cannot be composted or recycled in standard facilities."
}

# Item-specific reasoning
item_reasoning = {
    "Organic": "Natural food waste and plant material breaks down easily in compost.",
    "Wood": "Untreated wood is biodegradable and suitable for composting.",
    "Paper": "Clean paper products are biodegradable and can be composted.",
    "Paper bag": "Paper bags are biodegradable and compostable when not contaminated.",
    "Paper cups": "Paper cups without plastic lining can be composted.",
    "Cellulose": "Natural cellulose materials break down in compost environments.",
    
    "Cardboard": "Cardboard is made from paper fibers that can be recycled into new paper products.",
    "Glass bottle": "Glass can be melted down and reformed multiple times without quality degradation.",
    "Aluminum can": "Aluminum is infinitely recyclable and uses less energy than producing new aluminum.",
    "Plastic bottle": "Many plastic bottles (PET/HDPE) can be recycled into new plastic products.",
    "Plastic bag": "Clean plastic bags can be recycled at specialized facilities.",
    "Plastic cup": "Some plastic cups marked with recycle symbols can be processed at recycling centers.",
    "Plastic caps": "Hard plastic caps are often recyclable as #2, #4, or #5 plastics.",
    "Scrap metal": "Metal can be melted down and reused without losing quality.",
    "Tetra pack": "Multi-layer packaging that can be recycled through specialized processes.",
    "Aluminum caps": "Metal caps are recyclable similar to aluminum cans.",
    "Milk bottle": "Plastic milk bottles are typically HDPE (#2) which is widely recyclable.",
    
    "Aerosols": "Pressurized containers can be hazardous if not properly handled.",
    "Ceramic": "Ceramic doesn't break down and can contaminate recycling streams.",
    "Combined plastic": "Mixed plastic types are difficult to separate for recycling.",
    "Container for household chemicals": "May contain residual chemicals that contaminate recycling.",
    "Electronics": "Contains multiple materials and may require special e-waste processing.",
    "Foil": "Often contaminated with food waste making it unsuitable for standard recycling."
    # Additional items can be added as needed
}

# Store last detection results to prevent flickering
last_valid_results = []
last_update_time = time.time()
result_persistence_time = 2.0  # Keep results visible for at least 2 seconds

def get_category_and_reasoning(label):
    """Determine the category (compost, recyclable, garbage) and reasoning for a detected item."""
    for category, items in waste_categories.items():
        if label in items:
            general_reason = category_reasoning[category]
            specific_reason = item_reasoning.get(label, "")
            if specific_reason:
                reason = f"{specific_reason} {general_reason}"
            else:
                reason = general_reason
            return category, reason
    
    # Default to garbage if not found in categories
    return "garbage", "Unclassified items should be disposed of as garbage."

def format_results_html(results):
    """Format detection results as styled HTML for display."""
    if not results:
        return "<div class='no-detections'>No waste items detected</div>"
    
    html = "<div class='results-container'>"
    
    # Add summary count by category
    categories = {}
    for item in results:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    html += "<div class='summary-section'>"
    html += "<h3>Summary</h3>"
    for cat, count in categories.items():
        color_class = f"category-{cat}"
        html += f"<div class='category-count {color_class}'><span class='category-name'>{cat.upper()}</span>: {count} item{'s' if count > 1 else ''}</div>"
    html += "</div>"
    
    # Add detailed results
    html += "<div class='details-section'>"
    html += "<h3>Detected Items</h3>"
    
    for idx, item in enumerate(results):
        label = item["label"]
        category = item["category"]
        confidence = item["confidence"]
        reasoning = item["reasoning"]
        
        # Determine CSS class based on category
        category_class = f"category-{category}"
        
        html += f"""
        <div class='detection-item {category_class}'>
            <div class='detection-header'>
                <span class='item-number'>{idx + 1}</span>
                <span class='item-label'>{label}</span>
                <span class='item-category'>{category.upper()}</span>
                <span class='item-confidence'>Confidence: {confidence:.1%}</span>
            </div>
            <div class='detection-reasoning'>
                {reasoning}
            </div>
        </div>
        """
    
    html += "</div></div>"
    return html

def process_frame(frame):
    """Process a webcam frame and return the annotated frame with classification results."""
    global last_valid_results, last_update_time
    
    if frame is None:
        return None, last_valid_results
    
    # Process the frame with YOLO model
    results = model(frame)
    
    # Convert frame to PIL Image for drawing
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to load a larger font for better visibility
    try:
        title_font = ImageFont.truetype("arial.ttf", 30)  # Increased font size for better visibility
        label_font = ImageFont.truetype("arial.ttf", 22)  # Increased font size for details
    except IOError:
        # Fallback to default font if arial not available
        title_font = ImageFont.load_default()
        label_font = title_font
    
    # Color mapping for categories
    colors = {
        "compost": (0, 180, 0),    # Bright Green
        "recyclable": (0, 0, 255),  # Blue
        "garbage": (255, 0, 0)      # Red
    }
    
    detailed_results = []
    
    # Extract detections and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0].item())
            label = r.names[class_id]
            
            # Skip low confidence detections
            if conf < 0.35:  # Lowered threshold slightly
                continue
            
            # Get category and reasoning
            category, reasoning = get_category_and_reasoning(label)
            color = colors.get(category, (128, 128, 128))  # Default gray if category not found
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle with thicker line (width=4)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
            
            # Draw category in UPPERCASE with larger font
            category_text = f"{category.upper()}"
            label_text = f"{label} ({conf:.1%})"
            
            # First draw a semi-transparent background for the text
            # For the category (top label)
            text_width = title_font.getbbox(category_text)[2] if hasattr(title_font, 'getbbox') else 200
            text_height = title_font.getbbox(category_text)[3] if hasattr(title_font, 'getbbox') else 30
            draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)], 
                           fill=(*color, 180))  # Semi-transparent background
            
            # For the label (below category)
            label_width = label_font.getbbox(label_text)[2] if hasattr(label_font, 'getbbox') else 200
            label_height = label_font.getbbox(label_text)[3] if hasattr(label_font, 'getbbox') else 22
            draw.rectangle([(x1, y1), (x1 + label_width + 10, y1 + label_height + 5)], 
                           fill=(0, 0, 0, 180))  # Semi-transparent black
            
            # Now draw the text
            draw.text((x1 + 5, y1 - text_height - 5), category_text, font=title_font, fill=(255, 255, 255))
            draw.text((x1 + 5, y1 + 2), label_text, font=label_font, fill=(255, 255, 255))
            
            # Add to detailed results
            detailed_results.append({
                "label": label,
                "category": category,
                "confidence": conf,
                "reasoning": reasoning,
                "bounding_box": [x1, y1, x2, y2]
            })
    
    # Update the last valid results if we have any detections or enough time has passed
    current_time = time.time()
    if detailed_results:
        last_valid_results = detailed_results
        last_update_time = current_time
    elif (current_time - last_update_time) > result_persistence_time:
        # Only clear the results if enough time has passed without new detections
        last_valid_results = []
    
    # Use the most recent valid results (prevents flickering)
    results_to_return = last_valid_results if last_valid_results else detailed_results
    
    # Convert back to numpy array for Gradio
    annotated_frame = np.array(img)
    
    return annotated_frame, results_to_return

# Create Gradio interface
def create_ui():
    # Load CSS from file - with explicit direct path
    css_file_path = os.path.join("styles", "waste_classifier.css")
    
    try:
        with open(css_file_path, "r") as css_file:
            css = css_file.read()
        print(f"Successfully loaded CSS from {css_file_path}")
    except FileNotFoundError:
        print(f"CSS file not found at {css_file_path}, using default styles")
        # Create a minimal default CSS if file doesn't exist
        css = """
        .webcam-input { height: 550px !important; }
        .webcam-input button { z-index: 1000 !important; position: relative !important; }
        """
        # Try to save the CSS file
        try:
            with open(css_file_path, "w") as css_file:
                css_file.write(css)
            print(f"Created default CSS file at {css_file_path}")
        except Exception as e:
            print(f"Failed to create CSS file: {e}")

    with gr.Blocks(css=css, title="Smart Waste Classification") as demo:
        with gr.Column(elem_classes=["container"]):
            gr.Markdown("# Smart Waste Classification System", elem_classes=["app-title"])
            
            # Color-coded legend
            with gr.Row(elem_classes=["legend"]):
                with gr.Column(elem_classes=["legend-item"]):
                    gr.Markdown("<div><span class='legend-color compost-color'></span> COMPOST</div>")
                with gr.Column(elem_classes=["legend-item"]):
                    gr.Markdown("<div><span class='legend-color recyclable-color'></span> RECYCLABLE</div>")
                with gr.Column(elem_classes=["legend-item"]):
                    gr.Markdown("<div><span class='legend-color garbage-color'></span> GARBAGE</div>")
            
            # Main content with webcam and results
            with gr.Row():
                # Left column for webcam - with extra height for controls
                with gr.Column(scale=3, elem_classes=["left-column"]):
                    gr.Markdown("### Live Detection", elem_classes=["webcam-title"])
                    
                    # Webcam input with fixed height to ensure controls are visible
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        height=480, 
                        elem_classes=["webcam-input"],
                        label="Live Camera Feed",
                        interactive=True  # Make sure this is interactive
                    )
                
                # Right column for detailed results
                with gr.Column(scale=2):
                    gr.Markdown("### Classification Results", elem_classes=["results-title"])
                    # HTML output for formatted results
                    results_html = gr.HTML(elem_classes=["results-output"])
            
            # Hidden JSON output for processing
            results_json = gr.JSON(visible=False)
            
            # Stream the webcam input for real-time processing
            webcam_input.stream(
                fn=process_frame,
                inputs=webcam_input,
                outputs=[webcam_input, results_json],
                show_progress=False,
                time_limit=300,      # Process for up to 5 minutes at a time
                stream_every=0.1     # Process every 0.1 seconds for smooth real-time effect
            )
            
            # Update the HTML display whenever results_json changes
            results_json.change(
                fn=format_results_html,
                inputs=results_json,
                outputs=results_html
            )
            
            # How it works section
            with gr.Row(elem_classes=["how-it-works"]):
                gr.Markdown("""
                ## How It Works
                
                This system uses YOLOv8 to detect and classify waste items in real-time based on a dataset of common waste materials.
                
                The system categorizes items into three main groups:
                
                * **COMPOST** (Green): Biodegradable items that can break down naturally (organic waste, wood, paper)
                * **RECYCLABLE** (Blue): Items that can be processed and turned into new products (plastic bottles, glass, aluminum)
                * **GARBAGE** (Red): Items that cannot be composted or recycled in standard facilities
                
                Each detection includes the specific item type, classification category, confidence level, and an explanation of why it belongs in that category.
                """)
            
            gr.Markdown("### GarbEDGE - AI-Powered Waste Classification", elem_classes=["footer"])
    
    return demo

# Launch the app
if __name__ == "__main__":
    # Print current directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if CSS file exists
    css_path = os.path.join("styles", "waste_classifier.css")
    if os.path.exists(css_path):
        print(f"CSS file found at: {css_path}")
    else:
        print(f"CSS file not found at: {css_path}")
    
    demo = create_ui()
    # demo.launch()
    demo.launch(share=True)