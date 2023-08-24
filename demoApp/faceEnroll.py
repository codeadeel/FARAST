#!/usr/bin/env python3

'''
Face Enrollment App
===================

This App is used to enroll the faces using browser webcam
'''

# %%
# Importing Libraries
from inferenceGRPCClient import *
import os
from io import BytesIO
from PIL import Image
from bounding_box import bounding_box as bb
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from minio import Minio

# %%
# Basic Tools
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
    img = frame.to_ndarray(format="bgr24")
    if len(img)>0:
        img = ocv.flip(img, 1)
        imgDummy = img.copy()
        imgShape = img.shape
        ret = grpcInferenceClient([img], [detecThres], [nmsThres], [bxCus], [smFac], [recogThres], [fakeThres], [trFram])
        currentInfer = list(ret.values())[0]
        boxesList = np.array(currentInfer['bbox'][0])
        if len(boxesList)>0:
            bxAvailable = True
            boxesList[:, [0, 2]] *= imgShape[1]
            boxesList[:, [1, 3]] *= imgShape[0]
            boxesList = boxesList.astype(np.int16)
            bxArea = (boxesList[:, 2] - boxesList[:, 0]) * (boxesList[:, 3] - boxesList[:, 1])
            bBox = np.argmax(bxArea)
            cbx = boxesList[bBox]
            cla = currentInfer['faces'][0][bBox]
            if cla=='Small Face':
                cla = 'Come Closer!'
            try:
                if (cla=='Unknown'):
                    if (len(yourName)>0):
                        bb.add(img, cbx[0], cbx[1], cbx[2], cbx[3], yourName, 'blue')
                    else:
                        bb.add(img, cbx[0], cbx[1], cbx[2], cbx[3], 'Looking Sharp!', 'blue')
                else:
                    if (cla == 'Invalid Pose'):
                        bb.add(img, cbx[0], cbx[1], cbx[2], cbx[3], cla, 'red')
                    else:
                        bb.add(img, cbx[0], cbx[1], cbx[2], cbx[3], cla, 'orange')
            except:
                st.warning('Bad Frame Intercepted !!!', icon="⚠️")
            if ((enrl==True) and (cla=='Unknown')):
                faceBookCheck = [i.object_name.split('@')[0].lower() for i in bucketClient.list_objects(os.environ.get('facesTemplatesBucket', 'facebook'))]
                if len(yourName)==0:
                    ocv.putText(img, 'Please Write Your Name', (10, 20), ocv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, ocv.LINE_AA)
                elif yourName.lower() in faceBookCheck:
                    ocv.putText(img, f'{yourName} Already Exists !!!', (10, 20), ocv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, ocv.LINE_AA)
                elif (subPasscode != os.environ.get('submissionCode', 'abc')):
                    ocv.putText(img, 'Wrong Passcode !!!', (10, 20), ocv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, ocv.LINE_AA)
                else:
                    croppedimg = ocv.cvtColor(imgDummy[cbx[1]:cbx[3], cbx[0]:cbx[2], :], ocv.COLOR_BGR2RGB)
                    croppedimg = Image.fromarray(croppedimg)
                    byt1 = BytesIO()
                    croppedimg.save(byt1, format='JPEG')
                    client_name_chars = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
                    clientName = ''.join(np.random.choice(client_name_chars, size = 10).tolist())
                    bucketClient.put_object(
                        os.environ.get('facesTemplatesBucket', 'facebook'),
                        f'{yourName}@{clientName}.jpg',
                        BytesIO(byt1.getvalue()),
                        len(byt1.getvalue())
                    )
                    st.experimental_rerun()
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# %%
# Execution
@st.cache_resource(show_spinner=True)
def oneTimeRunner():
    bucketTemplate = os.environ.get('facesTemplatesBucket', 'facebook')
    fClient = faceClient(os.environ.get('serverIP', '0.0.0.0:4393'))
    bucketClient = Minio(
        os.environ.get('bucketUrl', '192.168.1.146:9000'),
        os.environ.get('bucketAccessKey', 'abc'),
        os.environ.get('bucketSecretKey', 'abc'),
        secure = True if os.environ.get('secureBucket', 'false').lower()=='true' else False
    )
    if not bucketClient.bucket_exists(bucketTemplate):
        bucketClient.make_bucket(bucketTemplate)
    return fClient, bucketClient

logoData = '/home/logo.png'
st.set_page_config(page_title="Face Detection, Recognition, Tracking", page_icon=logoData, layout="centered",initial_sidebar_state="collapsed")
grpcInferenceClient, bucketClient = oneTimeRunner()

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open(logoData))
with col2:
    st.header('Face Enrollment')
st.markdown("""---""")

st.sidebar.header('Face Tweaker')
st.sidebar.markdown("""---""")
detecThres = st.sidebar.slider('Detection Threshold', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
recogThres = st.sidebar.slider('Recognition Threshold', min_value=0.0, max_value=1.0, value=0.60, step=0.01)
fakeThres = st.sidebar.slider('Anti-Spoofing Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
nmsThres = st.sidebar.slider('Non-Max Suppression', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
smFac = st.sidebar.slider('Small Face Area Percentage', min_value=0.0, max_value=100.0, value=25.0, step=0.01)
bxCus = st.sidebar.number_input('Bounding Box Cushion', value=0)
trFram = st.sidebar.number_input('Tracking Frames', min_value=0, value=30)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

streamerRTC = webrtc_streamer(
    key = 'face',
    mode = WebRtcMode.SENDRECV,
    video_frame_callback = frameProcessor,
    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints = {'video': True, 'audio': False},
    async_processing = True
)

col3, col4 = st.columns(2)
with col3:
    yourName = st.text_input('Your Name', placeholder='Your Name', label_visibility='collapsed')
with col4:
    subPasscode = st.text_input('Submission Passcode', placeholder='Submission Passcode', type='password', label_visibility='collapsed')
enrl = st.button('Enroll', use_container_width=True)
