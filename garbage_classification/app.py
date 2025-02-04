import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your trained model path if applicable

st.title('Garbage Classification with YOLOv8')
st.write('Upload an image to detect and classify garbage items.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert('RGB')  # Ensure image is in RGB format
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform detection
    results = model.predict(image_np)
    print(results)

    # Display results
    st.write('Detected Objects:')
    if results:
        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image',
                    use_column_width=True)
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.data)
        except Exception as ex:
            # st.write(ex)
            st.write("No image is uploaded yet!")



        # annotated_frame = results[0].plot()  # Annotate the image with detections
        # st.image(annotated_frame, caption='Detected Objects', use_column_width=True)
    else:
        st.write('No objects detected.')