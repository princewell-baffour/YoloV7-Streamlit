import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor
from PIL import Image
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
import os


def yolov7():

    #st.header('AI Project - Object Detection')
    #st.subheader('YOLOv7 Model Trained on our Custom Dataset')
    #st.title('Options') 
    st.header('Project 4.0  - Yolov7 Model')
    st.subheader('YOLOv7 Model Trained on Custom Dataset(Strawberry)')
    st.write(" Made with ❤ by Princewell")
    # path to model 
    path_model_file = "models/yolov7best.pt" 


    source = ("Image Detection",  
    #"Video Detection",
    "WebCam")
    options = st.selectbox("Select input", range(
        len(source)), format_func=lambda x: source[x])


    # Confidence
    confidence = st.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)
   # Draw thickness
    draw_thick = st.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=5
    )

    # read class.txt
    class_labels = ('strawberry_flower',
                    'unripe_strawberry',
                    'duck',
                    'chicken',
                    'grape',
                    'watermelon') 


    # for i in range(len(class_labels)):
    #         classname = class_labels[i]

    # Image
    def ImageInput():
        image_file = st.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            
            file_bytes = np.asarray(
                bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            imga = Image.open(image_file)
            with col1:
                st.image(imga, caption='Uploaded Image', use_column_width='always')
            #FRAME_WINDOW.image(img, channels='BGR')

            model = custom(path_or_model=path_model_file)

            bbox_list = []
            current_no_class = []
            pred = model(img)
            box = pred.pandas().xyxy[0]
            class_list = box['class'].to_list()

            
        
            for i in box.index:
                xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                    int(box['ymax'][i]), box['confidence'][i]
                if conf > confidence:
                    bbox_list.append([xmin, ymin, xmax, ymax])
            if len(bbox_list) != 0:
                for bbox, id in zip(bbox_list, class_list):
                    plot_one_box(bbox, img, label=class_labels[id], line_thickness=draw_thick)
                    current_no_class.append([class_labels[id]])
            with col2:
                st.image(img, channels='BGR',caption='Model Prediction(s)')


            # Current number of classes
            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
            class_fq = json.dumps(class_fq, indent = 4)
            class_fq = json.loads(class_fq)
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
            
            # Updating Inference results
            with st.container():
                st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                st.markdown("<h3>Detected objects in current Frame</h3>", unsafe_allow_html=True)
                st.dataframe(df_fq, use_container_width=True)
                

    # #Video
    # def videoInput():
    #     p_time = 0
    #     upload_video_file = st.file_uploader(
    #         'Upload Video', type=['mp4', 'avi', 'mkv'])
    #     if upload_video_file is not None:
    #         # Model
    #         model = custom(path_or_model=path_model_file)

    #         tfile = tempfile.NamedTemporaryFile(delete=False)
    #         tfile.write(upload_video_file.read())
    #         cap = cv2.VideoCapture(tfile.name)
    #         FRAME_WINDOW.image([])
    #         stframe1 = st.empty()
    #         stframe2 = st.empty()
    #         stframe3 = st.empty()
    #         while True:
    #             success, img = cap.read()
    #             if not success:
    #                 st.error(
    #                     'Video file NOT working\n \
    #                     Check Video path or file properly!!',
    #                     icon="🚨"
    #                 )
    #                 break
    #             current_no_class = []
    #             bbox_list = []
    #             results = model(img)
    #             # Bounding Box
    #             box = results.pandas().xyxy[0]
    #             class_list = box['class'].to_list()

    #             for i in box.index:
    #                 xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
    #                     int(box['ymax'][i]), box['confidence'][i]
    #                 if conf > confidence:
    #                     bbox_list.append([xmin, ymin, xmax, ymax])
    #             if len(bbox_list) != 0:
    #                 for bbox, id in zip(bbox_list, class_list):
    #                     plot_one_box(bbox, img, label=class_labels[id], line_thickness=draw_thick)
    #                     current_no_class.append([class_labels[id]])
    #             FRAME_WINDOW.image(img, channels='BGR')
                
    #             # FPS
    #             c_time = time.time()
    #             fps = 1 / (c_time - p_time)
    #             p_time = c_time
                
    #             # Current number of classes
    #             class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
    #             class_fq = json.dumps(class_fq, indent = 4)
    #             class_fq = json.loads(class_fq)
    #             df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                
    #             # Updating Inference results
    #             with stframe1.container():
    #                 st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
    #                 if round(fps, 4)>1:
    #                     st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
    #                 else:
    #                     st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                
    #             with stframe2.container():
    #                 st.markdown("<h3>Detected objects</h3>", unsafe_allow_html=True)
    #                 st.dataframe(df_fq, use_container_width=True)

    #             with stframe3.container():
    #                 st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
    #                 js1, js2, js3 = st.columns(3)                       

    #                 # Updating System stats
    #                 with js1:
    #                     st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
    #                     mem_use = psutil.virtual_memory()[2]
    #                     if mem_use > 50:
    #                         js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
    #                     else:
    #                         js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

    #                 with js2:
    #                     st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
    #                     cpu_use = psutil.cpu_percent()
    #                     if mem_use > 50:
    #                         js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
    #                     else:
    #                         js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

    #                 with js3:
    #                     st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
    #                     js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)


    # Web-cam
    def WebCam():
        p_time = 0
        cam_options = st.selectbox('Webcam Channel',
                                        ('Select Channel', '0', '1', '2', '3'))
        # Model
        model = custom(path_or_model=path_model_file)

        sample_img = cv2.imread('yolologo.JPG')
        Webfeed = st.image(sample_img, channels='BGR')

        if len(cam_options) != 0:
            if not cam_options == 'Select Channel':
                cap = cv2.VideoCapture(int(cam_options))
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f'Webcam channel {cam_options} NOT working\n \
                            Change channel or Connect webcam properly!!',
                            icon="🚨"
                        )
                        break

                    bbox_list = []
                    current_no_class = []
                    results = model(img)
                    
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    #FRAME_WINDOW.image(img, channels='BGR')
                    Webfeed.image(img, channels='BGR',caption='Web Feed')

                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    
                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with stframe1.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        if round(fps, 4)>1:
                            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                    
                    with stframe2.container():
                        st.markdown("<h3>Detected objects in current Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq, use_container_width=True)

                    with stframe3.container():
                        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                        js1, js2, js3 = st.columns(3)                       

                        # Updating System stats
                        with js1:
                            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                            mem_use = psutil.virtual_memory()[2]
                            if mem_use > 50:
                                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

                        with js2:
                            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
                            cpu_use = psutil.cpu_percent()
                            if mem_use > 50:
                                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

                        with js3:
                            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
                            js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)

    

    if options == 0:    
        ImageInput()
    # elif options == 1:
    #     videoInput()
    else:
        WebCam()

