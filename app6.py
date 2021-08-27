import logging
from pathlib import Path
import streamlit as st
import pydub
import numpy as np
# import queue
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src import loadModel
import time

# from aiortc.contrib.media import MediaRecorder

from streamlit_webrtc import (
    # AudioProcessorBase,
    ClientSettings,
    # VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


HERE = Path(__file__).parent
logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": False,
        "audio": True,
    },
)

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')


# Load Model 
cnn = loadModel.CNN
cnn.model = cnn.loadTrainingModel()
classes = ['COPD-Mild', 'COPD-Severe', 'Interstitial Lung Disease', 'Normal']

