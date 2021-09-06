import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_VOICE_1 = os.path.join(DATA_DIR, 'yue_breathing_0823.aac')
DATA_DIR_VOICE_2 = os.path.join(DATA_DIR, 'COPD(mild)_ie_1_5_comp_0.10_vol_400.m4a')
DATA_DIR_VOICE_3 = os.path.join(DATA_DIR, 'COPD(severe)_ie_1_3.5_comp_0.08_vol_400.m4a')
DATA_DIR_VOICE_4 = os.path.join(DATA_DIR, 'interstitial_ie_1_2.8_comp_0.10_vol_250.m4a')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_H5 = os.path.join(MODEL_DIR, 'CNN_for4lungcondition_20210717.h5')

def saveWavFile(fn):    
    WAVE_OUTPUT_FILE = os.path.join(DATA_DIR, "{}.wav".format(fn))
    return WAVE_OUTPUT_FILE

def readWavFile(fn): #fn has Filename Extension(.wav)
    WAVE_OUTPUT_FILE = os.path.join(DATA_DIR, "{}".format(fn))
    return WAVE_OUTPUT_FILE


# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100   # Default sample rate of microphone or recording device
DURATION = 32   # 32 seconds
CHUNK_SIZE = 1024
