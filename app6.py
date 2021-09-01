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

from settings import DATA_DIR_VOICE_1, DATA_DIR_VOICE_2

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

        
def DoTheTest(fn, filepath):
    state_button_test1 = st.button(fn)
    if state_button_test1:
        #Load the sound data
        sound_data, sr = librosa.load(filepath, sr=44100)
        # st.text(sound_data)
        st.dataframe(sound_data)
        
        # Plot in time domain and frequency domain
        fig_place = st.empty()
        fig, [ax_time, ax_mfcc] = plt.subplots(nrows=2, ncols=1, gridspec_kw={"top": 1.5, "bottom": 0.2})
        fig.tight_layout()
        
        
        times = (np.arange(0, len(sound_data))) / sr
        ax_time.plot(times, sound_data)
        
        X = librosa.feature.mfcc(sound_data/1.0)       
        ax_mfcc.cla()
        img = librosa.display.specshow(X, x_axis='time')
        
        fig.colorbar(img, ax=ax_mfcc)
        ax_mfcc.set(title='MFCC')
        fig_place.pyplot(fig)
        
        data_pred = cnn.samplePred(cnn, sound_data)
        #st.text('data_pred: {}'.format(data_pred))
        
        #data_pred_class = np.argmax(np.round(data_pred), axis=1)
        #st.text('data_pred_class: {}'.format(data_pred_class))
        
        #s1 = classes[data_pred_class[0]]
        #s2 = np.round(float(data_pred[0,data_pred_class])*100, 4)
        #st.text("Predict class: {} for {}%".format(s1, s2))
        
        for i in range(len(classes)):
            st.text('{}: {}%'.format(classes[i],np.round(data_pred[0][i]*100,4)))
            
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def do_filter(y, N):
    y_filter = np.convolve(y, np.ones(N)/N, mode='same')
    return y_filter
    
def main():
    
    st.header("Classificaion for lung condition DEMO")
    '1. Please open the page using "Chrome" browser.'
    '2. Please prepare your microphone.'
    '3. Press "START" to breathe NORMALLY toward your microphone.'
    '4. Wait for 35 seconds to record your breathing sounds.'
    '5. Press "Result!!" to show the results. Have fun!!'
    
    "### Recording"
    
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1792, #256 = 5 seconds
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )
    
    if not webrtc_ctx.audio_receiver:
        st.info('Now condition: Stop recording.')
    
    # Test1 & Test
    DoTheTest('Example: Normal', DATA_DIR_VOICE_1)
    DoTheTest('Example: COPD(severe)', DATA_DIR_VOICE_2)
            
    if webrtc_ctx.audio_receiver:
        st.info('Now start recording.\n Please breathe toward the microphone.')
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
        
        
    
        state_button = st.button('Result!!')
        #Countdown
        if not state_button:
            countdown()
        
        if state_button:
            # try:
            # st.text('Click!')
            temp_sound_chunk = sound_chunk
            
            #preprocessing normalization
            sound_chunk = match_target_amplitude(sound_chunk, -20.0)  
            
            sound_chunk_mono = sound_chunk.set_channels(1) # Stereo to mono
            sample = np.array(sound_chunk_mono.get_array_of_samples())
            # sample = np.array(sound_chunk.get_array_of_samples())
            
            # Normalization 
            sample = sample/np.max(sample)
            # Filter
            sample = do_filter(sample, 10)
            
            fig_place = st.empty()
            fig, [ax_time, ax_mfcc] = plt.subplots(2,1, gridspec_kw={"top": 2.0, "bottom": 0.5})
            
            
            
            ax_time.cla()
            times = np.arange(0, len(sample)) / sound_chunk.frame_rate
            ax_time.plot(times, sample)
            
            
            # try:
            # X = librosa.load(sound_chunk)
            # XX = librosa.feature.mfcc(X)
            
            X = librosa.feature.mfcc(sample/1.0)
            # except:
                # st.error('Something wrong with librosa.feature.mfcc ...')
                
            ax_mfcc.cla()
            img = librosa.display.specshow(X, x_axis='time')
            fig.colorbar(img, ax=ax_mfcc)
            ax_mfcc.set(title='MFCC')
            
            fig_place.pyplot(fig)
            
            st.success('Success for plotting the data') 
            
            #Play the sounds
            # st.audio(sample)
            
            try:
                #Do Prediction
                data_pred = cnn.samplePred(cnn, sample/1.0)
                st.text('Results')
                for i in range(len(classes)):
                    st.text('{}: {}%'.format(classes[i],np.round(data_pred[0][i]*100,4)))
            except:
                st.error('Recording mest be over 35 seconds. Please press STOP and try again.')
                
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
