import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Real-Time Object Detection")

url = st.text_input("Enter Video URL", value="http://192.168.50.145:8080/video")

if st.button('Submit'):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
    else:
        st.success("Video stream opened successfully!")

    cap.set(3, 640)
    cap.set(4, 480)

    class_names = []
    class_file = 'coco.names'
    with open(class_file, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    frame_placeholder = st.empty()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.error("Error: Failed to read frame.")
            break

        class_ids, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, class_names[class_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        frame_placeholder.image(img_pil)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
