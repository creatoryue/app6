import logging
from pathlib import Path
import streamlit as st
import pydub
import numpy as np
# import queue
import matplotlib.pyplot as plt
#import librosa
#import librosa.display

# from src import loadModel
import time

# from aiortc.contrib.media import MediaRecorder

from streamlit_webrtc import (
    # AudioProcessorBase,
    ClientSettings,
    # VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

cnn = loadModel.CNN
cnn.model = cnn.loadTrainingModel(self=cnn)
classes = ['COPD-Mild', 'COPD-Severe', 'Interstitial Lung Disease', 'Normal']

