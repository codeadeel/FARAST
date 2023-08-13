#!/usr/bin/env python3

'''
Face Detection, Recognition & Tracking Live Demo App
====================================================

This App is used to run live demo using browser webcam
'''

# %%
# Importing Libraries
from inferenceGRPCClient import *
import os
from PIL import Image
from bounding_box import bounding_box as bb
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# %%
# Execution
@st.cache_resource(show_spinner=True)
def oneTimeRunner():
    return faceClient(os.environ.get('serverIP', '0.0.0.0:1234'))

logoData = '/home/logo.png'
st.set_page_config(page_title="Face Detection, Recognition, Tracking", page_icon=logoData, layout="centered",initial_sidebar_state="collapsed")
grpcInferenceClient = oneTimeRunner()
activer = False

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open(logoData))
with col2:
    st.header('Face Detection, Recognition & Tracking')
st.markdown("---")

rtspAddr = st.sidebar.text_input('RTSP Stream Address', placeholder='rtsp://')
if ((len(rtspAddr)>0) and (activer == False)):
    activer = True
    vid = ocv.VideoCapture(rtspAddr)

st.sidebar.header('Face Tweaker')
detecThres = st.sidebar.slider('Detection Threshold', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
recogThres = st.sidebar.slider('Recognition Threshold', min_value=0.0, max_value=1.0, value=0.60, step=0.01)
fakeThres = st.sidebar.slider('Anti-Spoofing Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
nmsThres = st.sidebar.slider('Non-Max Suppression', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
smFac = st.sidebar.slider('Small Face Area Percentage', min_value=0.0, max_value=100.0, value=1.0, step=0.01)
bxCus = st.sidebar.number_input('Bounding Box Cushion', value=0)
trFram = st.sidebar.number_input('Tracking Frames', min_value=0, value=30)

def frameProcessor(frame) -> av.VideoFrame:
    """
    This function serves as callback to process the subject frame from camera

    Arguments
    =========
    frame : Frame buffer received from the browser webcam

    Outputs
    =======
    Processed frame to display on browser
    """
    if activer:
        ret, img = vid.read()
        if ((len(img)==0) or (img==None)).any():
            img = frame.to_ndarray(format="bgr24")
    else:
        img = frame.to_ndarray(format="bgr24")
    if len(img)>0:
        img = ocv.flip(img, 1)
        imgShape = img.shape
        ret = grpcInferenceClient([img], [detecThres], [nmsThres], [bxCus], [smFac], [recogThres], [fakeThres], [trFram])
        currentInfer = list(ret.values())[0]
        boxesList = np.array(currentInfer['bbox'][0])
        if len(boxesList)>0:
            boxesList[:, [0, 2]] *= imgShape[1]
            boxesList[:, [1, 3]] *= imgShape[0]
            boxesList = boxesList.astype(np.int16)
            bxArea = (boxesList[:, 2] - boxesList[:, 0]) * (boxesList[:, 3] - boxesList[:, 1])
            bBox = np.argmax(bxArea)
            cRange = ['orange'] * len(bxArea)
            cRange[bBox] = 'blue'
            for b, l, c, r in zip(boxesList, currentInfer['faces'][0], cRange, currentInfer['spoofs'][0]):
                try:
                    if r=='Fake':
                        bb.add(img, b[0], b[1], b[2], b[3], l + ' : Fake', 'red')
                    else:
                        bb.add(img, b[0], b[1], b[2], b[3], l, c)
                except:
                    st.warning('Bad Frame Intercepted !!!', icon="⚠️")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key = 'face',
    mode = WebRtcMode.SENDRECV,
    video_frame_callback = frameProcessor,
    media_stream_constraints = {'video': True, 'audio': False},
    async_processing = True
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
