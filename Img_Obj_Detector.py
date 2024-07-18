import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Object Detection on Uploaded Image")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Display uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Convert image to BGR format (required by OpenCV)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load class names
    class_names = []
    class_file = 'coco.names'
    with open(class_file, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    # Load the neural network
    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Perform object detection
    class_ids, confs, bbox = net.detect(img_bgr, confThreshold=0.5)

    # Draw bounding boxes
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img_bgr, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img_bgr, class_names[class_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    # Convert the image back to RGB for displaying with PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Display the output image
    st.image(img_pil, caption='Processed Image.', use_column_width=True)
