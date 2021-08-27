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
cnn.model = cnn.loadTrainingModel(self=cnn)
classes = ['COPD-Mild', 'COPD-Severe', 'Interstitial Lung Disease', 'Normal']

def countdown():
    countdowntime = 35
    my_bar = st.progress(0)
    my_text = st.text('{}s'.format(0))

    start = time.time()
    for i in range(countdowntime):    
        time.sleep(1)
        end = time.time()
        # st.info('countdown for {} seconds'.format(countdowntime-np.round(end-start)))
        my_bar.progress(int(100/35*(end-start))-1)
        my_text.text('{}s'.format(countdowntime-np.round(end-start)))

def main():
    
    st.header("Classificaion for lung condition DEMO")
    '1. Please prepare your microphone.'
    '2. Press "START" to breathe NORMALLY toward your microphone.'
    '3. Wait for 35 seconds to record your breathing sounds.'
    '4. Press "Result!!" to show the results. Have fun!!'
    
    "### Recording"
    
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1792, #256 = 5 seconds
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )
    
    if not webrtc_ctx.audio_receiver:
        st.info('Now condition: Stop recording.')
        
        
    if webrtc_ctx.audio_receiver:
        st.info('Now strat recording.\n Please breathe toward the microphone.')
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except:
            logger.warning("Queue is empty. Abort.")
            st.error('ohoh')
            
        sound_chunk = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            sound_chunk += sound    
        
        #Countdown
        countdown()
    
        state_button = st.button('Result!!')
        if state_button:
            # try:
            # st.text('Click!')
            sound_chunk = sound_chunk.set_channels(1) # Stereo to mono
            sample = np.array(sound_chunk.get_array_of_samples())
            
            
            fig_place = st.empty()
            fig, [ax_time, ax_mfcc] = plt.subplots(2,1)
            
            ax_time.cla()
            times = (np.arange(-len(sample), 0)) / sound_chunk.frame_rate
            ax_time.plot(times, sample)
            
            
            # try:
            X = librosa.feature.mfcc(sample/1.0)
            # except:
                # st.error('Something wrong with librosa.feature.mfcc ...')
                
            ax_mfcc.cla()
            librosa.display.specshow(X, x_axis='time')
            fig_place.pyplot(fig)
            st.success('PLotting the data...') 
            
            
            #Do Prediction
            data_pred = cnn.samplePred(cnn, sample/1.0)
            data_pred_class = np.argmax(np.round(data_pred), axis=1)
    
            
            s1 = classes[data_pred_class[0]] # s2 is the number of the classes
            s2 = np.round(float(data_pred[0,data_pred_class])*100, 4) # s1 is the percentage of the predicted class
            st.text("Predict class: {} for {}%".format(s1, s2))

if __name__ == '__main__':
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
    
    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    
    main()
