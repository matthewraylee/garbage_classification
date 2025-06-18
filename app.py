import os
import gradio as gr
from garbage_classification.waste_classification_webcam import create_ui

# Create and launch the interface
app = create_ui()

# Launch the app (HF Spaces will expose port 7860)
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
