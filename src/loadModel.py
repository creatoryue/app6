from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras import models
from tensorflow.keras.models import load_model

# import librosa
import numpy as np
from settings import MODEL_H5

n_timesteps = 1290
input_shape = (1290, 20)

class CNN(object):
    def __init__(self, most_shape):
        self.model = models.Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(30), activation='relu')
        self.model.add(Dropout(rate=0.3))

        self.model.add(Dense(4, activation='softmax'))
        opt = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        
    def __str__(self):
        return str(self.model.summary())

    def loadTrainingModel(self):
        '''load the CNN model'''
        # self.model = models.load_model('.\\models\\model.h5')
        # self.model = models.load_model('CNN_for4lungcondition_20210717.h5')
        # self.model = models.load_model(MODEL_H5)
        self.model = load_model(MODEL_H5)
        
        return self.model
    
    def Hello(self):
        print("Hello my friend")
        
    def Summary(self):
        '''Show model summary'''
        self.model.summary()
        
    def samplePred(self, data):
        # data, sampling_rate = librosa.load(path_test+'\\'+ filename)
        # X = librosa.feature.mfcc(data)
        XX = X[:, 0:n_timesteps]
        XX = XX.T[np.newaxis, ...]
        #XX.shape
        data_pred = self.model.predict(XX)
        # data_pred = np.argmax(np.round(data_pred), axis=1)
        
        return data_pred
