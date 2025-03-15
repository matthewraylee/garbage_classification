import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time

# Create necessary directories
os.makedirs("styles", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Load trained YOLO model
# model_dir = "runs/detect/train24/weights/best.pt"  # Change this to your model path
model_dir = os.getenv("MODEL_PATH", "runs/detect/train24/weights/best.pt")
model = YOLO(model_dir)

# Define waste categories with clear organization
WASTE_CATEGORIES = {
    "compost": {
        "color": (0, 180, 0),  # Bright Green
        "items": ["Organic", "Wood", "Paper", "Paper bag", "Paper cups", "Cellulose"],
        "description": "Biodegradable items that break down naturally"
    },
    "recyclable": {
        "color": (0, 0, 255),  # Blue
        "items": [
            "Cardboard", "Glass bottle", "Aluminum can", "Plastic bottle", "Plastic bag",
            "Plastic cup", "Plastic caps", "Scrap metal", "Paper", "Tetra pack", 
            "Aluminum caps", "Milk bottle"
        ],
        "description": "Items that can be processed and reused"
    },
    "garbage": {
        "color": (255, 0, 0),  # Red
        "items": [
            "Aerosols", "Ceramic", "Combined plastic", "Container for household chemicals",
            "Disposable tableware", "Electronics", "Foil", "Furniture", "Iron utensils",
            "Liquid", "Metal shavings", "Paper shavings", "Papier mache", "Plastic can",
            "Plastic canister", "Plastic shaker", "Plastic shavings", "Plastic toys",
            "Postal packaging", "Printing industry", "Stretch film", "Textile", "Tin",
            "Unknown plastic", "Zip plastic bag"
        ],
        "description": "Items that cannot be composted or recycled in standard facilities"
    }
}

# Detailed reasoning for each category
CATEGORY_REASONING = {
    "compost": "This item is biodegradable and can be broken down naturally into compost.",
    "recyclable": "This item can be processed and reused to make new products.",
    "garbage": "This item cannot be composted or recycled in standard facilities."
}

# Item-specific reasoning with more comprehensive explanations
ITEM_REASONING = {
    # Compost items
    "Organic": "Natural food waste and plant material breaks down easily in compost.",
    "Wood": "Untreated wood is biodegradable and suitable for composting.",
    "Paper": "Clean paper products are biodegradable and can be composted.",
    "Paper bag": "Paper bags are biodegradable and compostable when not contaminated.",
    "Paper cups": "Paper cups without plastic lining can be composted.",
    "Cellulose": "Natural cellulose materials break down in compost environments.",
    
    # Recyclable items
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
    
    # Garbage items
    "Aerosols": "Pressurized containers can be hazardous if not properly handled.",
    "Ceramic": "Ceramic doesn't break down and can contaminate recycling streams.",
    "Combined plastic": "Mixed plastic types are difficult to separate for recycling.",
    "Container for household chemicals": "May contain residual chemicals that contaminate recycling.",
    "Electronics": "Contains multiple materials and may require special e-waste processing.",
    "Foil": "Often contaminated with food waste making it unsuitable for standard recycling."
    # Additional items can be added as needed
}

# Detection persistence settings
last_valid_results = []
last_update_time = time.time()
result_persistence_time = 2.0  # Keep results visible for at least 2 seconds

def get_category_info(label):
    """Determine the category and reasoning for a detected item."""
    # Find which category contains this item
    for category, data in WASTE_CATEGORIES.items():
        if label in data["items"]:
            general_reason = CATEGORY_REASONING[category]
            specific_reason = ITEM_REASONING.get(label, "")
            
            # Combine reasons if we have both
            if specific_reason:
                reason = f"{specific_reason} {general_reason}"
            else:
                reason = general_reason
                
            return {
                "category": category,
                "color": data["color"],
                "reason": reason
            }
    
    # Default to garbage if not found in categories
    return {
        "category": "garbage",
        "color": WASTE_CATEGORIES["garbage"]["color"],
        "reason": "Unclassified items should be disposed of as garbage."
    }

def format_results_html(results):
    """Format detection results as styled HTML for display."""
    if not results:
        return """
        <div class='no-detections'>
            <div class='empty-state'>
                <div class='empty-icon'>📷</div>
                <div class='empty-message'>Point your camera at waste items to classify them</div>
            </div>
        </div>
        """
    
    html = "<div class='results-container'>"
    
    # Add summary count by category
    categories = {}
    for item in results:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    html += "<div class='summary-section'>"
    html += "<h3>Summary</h3>"
    html += "<div class='category-counts'>"
    
    # Create summary bars for each category
    for category in ["compost", "recyclable", "garbage"]:
        count = categories.get(category, 0)
        total = len(results)
        percentage = (count / total * 100) if total > 0 else 0
        
        html += f"""
        <div class='category-summary'>
            <div class='category-label category-{category}'>
                <span class='category-icon'></span>
                <span class='category-name'>{category.upper()}</span>
            </div>
            <div class='category-bar-container'>
                <div class='category-bar category-{category}-bg' style='width: {percentage}%;'></div>
                <span class='category-count'>{count}</span>
            </div>
        </div>
        """
    
    html += "</div></div>"  # Close summary section
    
    # Add detailed results
    html += "<div class='details-section'>"
    html += "<h3>Detected Items</h3>"
    html += "<div class='items-list'>"
    
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
                <div class='item-info'>
                    <span class='item-number'>{idx + 1}</span>
                    <span class='item-label'>{label}</span>
                </div>
                <div class='item-meta'>
                    <span class='item-category'>{category.upper()}</span>
                    <span class='item-confidence'>{confidence:.1%}</span>
                </div>
            </div>
            <div class='detection-reasoning'>
                {reasoning}
            </div>
        </div>
        """
    
    html += "</div></div></div>"  # Close details and container
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
        # Try to use a more readable font if available
        title_font = ImageFont.truetype("arial.ttf", 30)
        label_font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        # Fallback to default font if arial not available
        title_font = ImageFont.load_default()
        label_font = title_font
    
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
            if conf < 0.35:
                continue
            
            # Get category info
            category_info = get_category_info(label)
            category = category_info["category"]
            color = category_info["color"]
            reasoning = category_info["reason"]
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle with thicker line
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
            
            # Prepare text
            category_text = f"{category.upper()}"
            label_text = f"{label} ({conf:.1%})"
            
            # Calculate text dimensions
            text_width = title_font.getbbox(category_text)[2] if hasattr(title_font, 'getbbox') else 200
            text_height = title_font.getbbox(category_text)[3] if hasattr(title_font, 'getbbox') else 30
            
            # Draw background for category label (top)
            draw.rectangle(
                [(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)], 
                fill=(*color, 180)  # Semi-transparent background
            )
            
            # Draw background for item label (bottom)
            label_width = label_font.getbbox(label_text)[2] if hasattr(label_font, 'getbbox') else 200
            label_height = label_font.getbbox(label_text)[3] if hasattr(label_font, 'getbbox') else 22
            
            draw.rectangle(
                [(x1, y1), (x1 + label_width + 10, y1 + label_height + 5)], 
                fill=(0, 0, 0, 180)  # Semi-transparent black
            )
            
            # Draw text
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

def create_css():
    """Create an improved CSS file for the application."""
    css = """
    /* Global Styles */
    :root {
        --compost-color: rgb(0, 180, 0);
        --recyclable-color: rgb(0, 0, 255);
        --garbage-color: rgb(255, 0, 0);
        --background-color: #f5f7fa;
        --card-background: white;
        --text-color: #333;
        --border-radius: 8px;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Container and Layout */
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    .app-title {
        text-align: center;
        margin-bottom: 20px;
        color: #2d3748;
        font-size: 32px;
    }

    /* Legend Styling */
    .legend {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        padding: 10px;
        box-shadow: var(--shadow);
    }

    .legend-item {
        display: flex;
        align-items: center;
        margin: 0 15px;
        font-weight: bold;
    }

    .legend-color {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 8px;
        border-radius: 4px;
    }

    .compost-color {
        background-color: var(--compost-color);
    }

    .recyclable-color {
        background-color: var(--recyclable-color);
    }

    .garbage-color {
        background-color: var(--garbage-color);
    }

    /* Webcam Container Styling */
    .webcam-container {
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
        height: 480px;
    }

    /* Fix webcam input styling */
    .webcam-input {
        height: 100% !important;
        width: 100% !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    .webcam-input > div {
        height: 100% !important;
    }

    /* Style the webcam button */
    .webcam-input button {
        position: absolute !important;
        bottom: 20px !important;
        left: 20px !important;
        z-index: 1000 !important;
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 24px !important;
        border: 2px solid white !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    .webcam-input button:hover {
        background-color: rgba(0, 0, 0, 0.9) !important;
        transform: scale(1.05) !important;
    }

    /* Style the webcam placeholder */
    .webcam-placeholder {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: #718096;
    }

    .webcam-placeholder img {
        width: 64px;
        height: 64px;
        margin-bottom: 16px;
        opacity: 0.7;
    }

    .webcam-placeholder p {
        font-size: 16px;
        margin: 0;
    }

    .webcam-title, .results-title {
        margin-bottom: 10px;
        font-size: 20px;
        color: #4a5568;
    }

    /* Results Styling */
    .results-output {
        height: 480px;
        overflow-y: auto;
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        padding: 15px;
        box-shadow: var(--shadow);
    }

    .no-detections {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #718096;
    }

    .empty-state {
        text-align: center;
    }

    .empty-icon {
        font-size: 48px;
        margin-bottom: 10px;
    }

    .empty-message {
        font-size: 16px;
    }

    .summary-section, .details-section {
        margin-bottom: 20px;
    }

    .summary-section h3, .details-section h3 {
        font-size: 18px;
        margin-bottom: 10px;
        color: #4a5568;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 5px;
    }

    .category-counts {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .category-summary {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .category-label {
        display: flex;
        align-items: center;
        width: 120px;
        font-weight: bold;
    }

    .category-icon {
        display: inline-block;
        width: 12px;
        height: 12px;
        margin-right: 5px;
        border-radius: 50%;
    }

    .category-compost .category-icon {
        background-color: var(--compost-color);
    }

    .category-recyclable .category-icon {
        background-color: var(--recyclable-color);
    }

    .category-garbage .category-icon {
        background-color: var(--garbage-color);
    }

    .category-bar-container {
        flex: 1;
        height: 20px;
        background-color: #edf2f7;
        border-radius: 10px;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
    }

    .category-bar {
        height: 100%;
        transition: width 0.3s ease;
    }

    .category-compost-bg {
        background-color: var(--compost-color);
    }

    .category-recyclable-bg {
        background-color: var(--recyclable-color);
    }

    .category-garbage-bg {
        background-color: var(--garbage-color);
    }

    .category-count {
        position: absolute;
        right: 10px;
        color: #2d3748;
        font-weight: bold;
    }

    .items-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .detection-item {
        background-color: #f8fafc;
        border-radius: var(--border-radius);
        padding: 12px;
        border-left: 5px solid #cbd5e0;
    }

    .category-compost {
        border-left-color: var(--compost-color);
    }

    .category-recyclable {
        border-left-color: var(--recyclable-color);
    }

    .category-garbage {
        border-left-color: var(--garbage-color);
    }

    .detection-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .item-info {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .item-number {
        background-color: #e2e8f0;
        color: #4a5568;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }

    .item-label {
        font-weight: bold;
        font-size: 16px;
    }

    .item-meta {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .item-category {
        font-size: 12px;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 12px;
        color: white;
    }

    .category-compost .item-category {
        background-color: var(--compost-color);
    }

    .category-recyclable .item-category {
        background-color: var(--recyclable-color);
    }

    .category-garbage .item-category {
        background-color: var(--garbage-color);
    }

    .item-confidence {
        font-size: 12px;
        color: #718096;
    }

    .detection-reasoning {
        font-size: 14px;
        color: #4a5568;
        line-height: 1.4;
    }

    /* How it works section */
    .how-it-works {
        margin-top: 30px;
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--shadow);
    }

    .how-it-works h2 {
        color: #2d3748;
        margin-bottom: 15px;
    }

    .how-it-works ul {
        margin-left: 20px;
    }

    .footer {
        text-align: center;
        margin-top: 20px;
        color: #718096;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .legend {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .legend-item {
            margin: 5px 0;
        }
        
        .detection-header {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .item-meta {
            margin-top: 5px;
        }
    }
    """
    
    # Save CSS to file
    css_file_path = os.path.join("styles", "waste_classifier.css")
    with open(css_file_path, "w") as css_file:
        css_file.write(css)
    
    return css

# Create Gradio interface
def create_ui():
    # Create and load CSS
    css = create_css()
    
    with gr.Blocks(css=css, title="Smart Waste Classification") as demo:
        with gr.Column(elem_classes=["container"]):
            gr.Markdown("# GarbEDGE: Smart Waste Classification", elem_classes=["app-title"])
            
            # Color-coded legend
            with gr.Row(elem_classes=["legend"]):
                for category, data in WASTE_CATEGORIES.items():
                    with gr.Column(elem_classes=["legend-item"]):
                        gr.Markdown(f"<div><span class='legend-color {category}-color'></span> {category.upper()}</div>")
                        gr.Markdown(f"<div class='legend-description'>{data['description']}</div>")
            
            # Main content with webcam and results
            with gr.Row():
                # Left column for webcam
                with gr.Column(scale=3):
                    gr.Markdown("### Live Detection", elem_classes=["webcam-title"])
            
                    # Use a Column instead of Box (which doesn't exist in Gradio)
                    with gr.Column(elem_classes=["webcam-container"]):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            elem_classes=["webcam-input"],
                            label=None,  # Remove label as we have the title above
                            interactive=True
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
                
                ### Instructions
                
                1. Allow camera access when prompted
                2. Point your camera at waste items
                3. View real-time classifications on the right panel
                4. Use the information to properly sort your waste
                """)
            
            gr.Markdown("### GarbEDGE - AI-Powered Waste Classification", elem_classes=["footer"])
    
    return demo

# Launch the app
if __name__ == "__main__":
    print("Starting GarbEDGE Waste Classification System...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if CSS file exists
    css_path = os.path.join("styles", "waste_classifier.css")
    if os.path.exists(css_path):
        print(f"CSS file found at: {css_path}")
    else:
        print(f"CSS file will be created at: {css_path}")
    
    demo = create_ui()
    # demo.launch(share=True)  # Set share=True to create a public link
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

    
    print("Application started successfully!")