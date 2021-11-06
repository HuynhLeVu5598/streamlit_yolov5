from typing import List

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from config import CLASSES1, CLASSES2, WEBRTC_CLIENT_SETTINGS

html = """
<div style = "background-color:black;padding:18px">
<h1 style = "color:green; text-align:center"> Detect Object </h1>
</div>
"""
st.set_page_config(
    page_title="YOLOv5 demo",
)

st.markdown(html, unsafe_allow_html =True)

@st.cache(max_entries=2)
def get_yolo5(weights):
    return torch.hub.load('./yolov5','custom',path = '{}'.format(weights),source ='local', force_reload =True)


@st.cache(max_entries=10)
def get_preds(img, imgsz):
    if all_classes == False:
        result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = classes)
    elif all_classes == True:
        result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
    return result.xyxy[0].numpy()


def get_colors(indexes):
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids

    def get_preds(self, img, imgsz):
        if all_classes == False:
            result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = classes)
        elif all_classes == True:
            result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
        return result.xyxy[0].numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.get_preds(img, imgsz)

        result = result[np.isin(result[:,-1], self.target_class_ids)]
        
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            st.write(label)
            img = cv2.rectangle(img, p0, p1, self.rgb_colors[label], 2) 
            img = cv2.putText(img, CLASSES[label], (p0[0], p0[1]),  cv2.FONT_HERSHEY_SIMPLEX, 2, self.rgb_colors[label],2, lineType=cv2.LINE_AA)   

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

st.markdown("""
        <style>
        .format-style{
            font-size:20px;
            font-family:arial;
            color:yellow;
        }
        </style>
        """,
        unsafe_allow_html= True
    )

st.markdown(
    """
    <style>.
    common-style{
        font-size:18px;
        font-family:arial;
        color:pink;
    }
    </style>
    """,
    unsafe_allow_html= True
)
st.sidebar.markdown(
    '<p class = "format-style"> Parameter </p>',
    unsafe_allow_html= True
)


weights = st.sidebar.selectbox(
    'Weights', 
    ('yolov5s.pt','best_bt.pt'),
    format_func = lambda a: a[:len(a)-3] 
)

if weights == 'yolov5s.pt':
    CLASSES = CLASSES1
elif weights == 'best_bt.pt':
    CLASSES = CLASSES2


imgsz = st.sidebar.selectbox(
    'Size Image',
    (416,512,608,896,1024,1280,1408,1536)
)

conf_thres = st.sidebar.slider(
    'Confidence Threshold', 0.00, 1.00, 0.7
)

iou_thres = st.sidebar.slider(
    'IOU Threshold', 0.00,1.00, 0.45
)
max_det = st.sidebar.selectbox(
    'Max detection',
    [i for i in range(1,20)]
)

classes = st.sidebar.multiselect(
    'Classes',
    [i for i in range(len(CLASSES))],
    format_func= lambda index: CLASSES[index]
)

all_classes = st.sidebar.checkbox('All classes', value =False)



with st.spinner('Loading the model...'):
    model = get_yolo5(weights)

st.success('Loading the model.. Done!')

prediction_mode = st.sidebar.radio(
    "",
    ('Single image', 'Web camera'),
    index=0)

if all_classes:
    target_class_ids = list(range(len(CLASSES)))
elif classes:
    target_class_ids = [class_name for class_name in classes]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)

detected_ids = None


if prediction_mode == 'Single image':
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img,imgsz)

        result_copy = result.copy()

        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
     

        detected_ids = []
        img_draw = img.copy().astype(np.uint8)

        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img_draw = cv2.rectangle(img_draw, 
                                    p0, p1, 
                                    rgb_colors[label], 2) 
            detected_ids.append(label)
        
        st.image(img_draw, use_column_width=True)

elif prediction_mode == 'Web camera':
    ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,)

    if ctx.video_transformer:
        ctx.video_transformer.model = model
        ctx.video_transformer.rgb_colors = rgb_colors
        ctx.video_transformer.target_class_ids = target_class_ids


