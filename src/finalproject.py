'''
Classification of Breathing Sound into 4 classes
'''
import os
import numpy as np

path_test = os.getcwd()
classes = ['COPD-Mild', 'COPD-Severe', 'Interstitial Lung Disease', 'Normal']
print(classes)
classes_num = list(range(len(classes)+1))
labels = dict(zip(classes,classes_num))

lr = 1e-3




# In[] Load Training Data
import librosa

y_labels_train=[]
x_raw_train=[]

for f in classes:
    filenames = os.listdir('{}\\{}\\{}\\{}'.format(path_test,'sound data', 'TrainData', f))
#     filenames = random.sample(fnames,10)
    if len(filenames)>0:
        for filename in filenames:
            data, sampling_rate = librosa.load(path_test+'\\'+'sound data'+
                                               '\\'+'TrainData'+'\\'+f+'\\'+filename)
            X = librosa.feature.mfcc(data, sr=sampling_rate)
            y_labels_train.append(f)
            x_raw_train.append(X.T)

n_timesteps = np.min([len(v) for v in x_raw_train])

y = np.empty(len(x_raw_train), dtype=int)
x_train = []

for i in range(len(x_raw_train)):
    y[i] = labels[y_labels_train[i]]
    if len(x_raw_train[i])>n_timesteps:
        x_raw_train[i] = x_raw_train[i][:n_timesteps]
    x_train.append(x_raw_train[i])
x_train = np.asarray(x_train)                      
Y_train = y
n_features = x_train.shape[2]


# In[] Load Testing Data
import librosa

y_labels_test =[]
x_raw_test=[]

for f in classes:
    filenames = os.listdir('{}\\{}\\{}\\{}'.format(path_test,'sound data', 'TestData', f))
#     filenames = random.sample(fnames,10)
    if len(filenames)>0:
        for filename in filenames:
            data, sampling_rate = librosa.load(path_test+'\\'+'sound data'+'\\'+'TestData'+'\\'+f+'\\'+filename)
            X = librosa.feature.mfcc(data, sr=sampling_rate)
            y_labels_test.append(f)
            x_raw_test.append(X.T)

n_timesteps = np.min([len(v) for v in x_raw_train])#x_raw_train
                     
y = np.empty(len(x_raw_test), dtype=int)
x_test = []

for i in range(len(x_raw_test)):
    y[i] = labels[y_labels_test[i]]
    if len(x_raw_test[i])>n_timesteps:
        x_raw_test[i] = x_raw_test[i][:n_timesteps]
    x_test.append(x_raw_test[i])
x_test = np.asarray(x_test)                      
Y_test = y
n_features = x_test.shape[2]
# In[]
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load(path_test+'\\'+'sound data'+'\\'+'TrainData'+'\\'+'Normal'+'\\'+'normal_ie_1_2_comp_0.08_vol_500.m4a')
t = np.linspace(0, len(data)/sampling_rate, len(data))
plt.figure(1)
plt.plot(t,data)
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.title('Breathing sounds for normal condition')

data, sampling_rate = librosa.load(path_test+'\\'+'sound data'+'\\'+'TrainData'+'\\'+'COPD-Severe'+'\\'+'COPD(severe)_ie_1_3_comp_0.10_vol_500.m4a')
t = np.linspace(0, len(data)/sampling_rate, len(data))
plt.figure(2)
plt.plot(t,data)
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.title('Breathing sounds for severe COPD condition')

data, sampling_rate = librosa.load(path_test+'\\'+'voice_20309.aac')
t = np.linspace(0, len(data)/sampling_rate, len(data))
plt.figure(3)
plt.plot(t,data)
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.title('Breathing sounds for my lung condition')

# In[]
import matplotlib.pyplot as plt
import librosa.display

for f in classes:
    filenames = os.listdir('{}\\{}\\{}\\{}'.format(path_test,'sound data', 'TrainData', f))
    data, sampling_rate = librosa.load(path_test+'\\'+'sound data'+'\\'+'TrainData'+'\\'+f+'\\'+filenames[0])
    X = librosa.feature.mfcc(data, sr=sampling_rate)
    fig = plt.figure()
    librosa.display.specshow(X, x_axis='time')
    plt.title(f)
    plt.show()
    
# In[] model1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


input_shape =(n_timesteps, n_features)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv1D(filters=20, kernel_size=3,strides=5,
                      activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01),
                      bias_regularizer=keras.regularizers.l2(0.01),
                      input_shape =(n_timesteps, n_features)),
        layers.MaxPooling1D(pool_size=3),
        
#        layers.Conv1D(filters=10, kernel_size=3,strides=5,
#                      activation='relu',
#                      kernel_regularizer=keras.regularizers.l2(0.01),
#                      bias_regularizer=keras.regularizers.l2(0.01),
#                      input_shape =(n_timesteps, n_features)),
                      
        layers.Conv1D(filters=5, kernel_size=3,strides=5,
                      activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01),
                      bias_regularizer=keras.regularizers.l2(0.01),
                      input_shape =(n_timesteps, n_features)),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(len(classes), activation="softmax"),
    ]
)
opt = keras.optimizers.Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# In[] model2
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential,Model,load_model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf

input_shape =(n_timesteps, n_features)

model = Sequential()
model.add(Conv1D(filters=5, kernel_size=5,
                 strides=5, activation='relu'
                 , kernel_regularizer=l2(0.01),
                 bias_regularizer=l2(0.01),
                 input_shape =(n_timesteps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv1D(filters=20, kernel_size=5,
                 strides=5, activation='relu',
                 kernel_regularizer=l2(0.01),
                 bias_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(30, activation='relu',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)))
model.add(Dense(len(classes), activation='softmax'))
opt = Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# In[] model3
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models

input_shape =(n_timesteps, n_features)


def CreateModel(input_shape):
    # create model
    model = models.Sequential()
    
    # 1st conv layer
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool1D(3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    
    # 2nd conv layer
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPool1D(3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())

    # 3rd conv layer
    model.add(layers.Conv1D(32, kernel_size=2, activation='relu'))
    model.add(layers.MaxPool1D(2, strides=2, padding='same'))
    model.add(layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # output layer
    model.add(layers.Dense(len(classes), activation='softmax'))

    return model

if __name__ == "__main__":
    # Create CNN model
    input_shape = (n_timesteps, n_features)
    model = CreateModel(input_shape)
    
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=lr) 
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    model.summary()

# In[] model4
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential,Model,load_model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
from keras import layers
from keras import models

input_shape =(n_timesteps, n_features)


model = models.Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
model.add(layers.BatchNormalization())

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
model.add(layers.BatchNormalization())

model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(layers.BatchNormalization())

model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(classes), activation='softmax'))
opt = Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

# In[]

model.load_weights('CNNmodelMFCC300_2000.h5', by_name=False, skip_mismatch=False)


# In[]
import keras

batch_size = 128
epochs = 100

Y_train_C = keras.utils.to_categorical(Y_train, len(classes))
Y_test_C = keras.utils.to_categorical(Y_test, len(classes))

# In[]
from tensorflow.keras.models import load_model
history = model.fit(x_train, Y_train_C, batch_size=batch_size, epochs=epochs)

model.save('CNN_for4lungcondition_20210626.h5')
model.save('model.h5')
# In[]
#model.save('DLMIA_CNN_0613_1.h5')
import keras
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()
# In[]evaluate the model

test_error, test_accuracy = model.evaluate(x_test, Y_test_C, verbose=1)
print("Test error: {}, and Test acc: {}".format(test_error, test_accuracy))

# In[]
from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) # Convert one-hot to index
Y_test = y #y=Y_test

print(classification_report(Y_test, y_pred, target_names=np.asarray(classes)))

# In[]
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = confusion_matrix(Y_test, y_pred)

# Plot normalized confusion matrix
fig = plt.figure()
fig.set_size_inches(10, 8, forward=True)
plt.rcParams.update({'font.size': 16})
plot_confusion_matrix(cnf_matrix, classes=np.asarray(classes), normalize=True,
                      title='CNN Normalized Confusion Matrix')

# In[] Sample predition
import librosa

data, sampling_rate = librosa.load(path_test+'\\'+'COPD(severe)_ie_1_3.5_comp_0.08_vol_400.m4a')
X = librosa.feature.mfcc(data, sr=sampling_rate)
XX = X[:, 0:n_timesteps]

XX = XX.T[np.newaxis, ...]
#XX.shape
data_pred = model.predict(XX)
data_pred = np.argmax(np.round(data_pred), axis=1)
print('Predict class: {}'.format(classes[data_pred[0]]))

def samplePred(model, filename):
    data, sampling_rate = librosa.load(path_test+'\\'+ filename)
    X = librosa.feature.mfcc(data, sr=sampling_rate)
    XX = X[:, 0:n_timesteps]
    XX = XX.T[np.newaxis, ...]
    #XX.shape
    data_pred = model.predict(XX)
    data_pred = np.argmax(np.round(data_pred), axis=1)
    print('Predict class: {}'.format(classes[data_pred[0]]))
    return data_pred

fn = 'COPD(severe)_ie_1_4.5_comp_0.09_vol_400.m4a'
samplePred(model, fn)



from sklearn.metrics import classification_report
print(classification_report(np.array([0,0,0,1]), data_pred, target_names=np.asarray(classes)))

#
#def samplepredict(model, data):
#    data = data[np.newaxis ,...]    
#    data_pred = model.predict(data)
#    data_pred = np.argmax(data_pred, axis=1) # Convert one-hot to index
#    
#    print(classification_report(data, data_pred, target_names=np.asarray(classes)))
#
#samplepredict(model, XX.T)


# In[]
model.save('CNN_for4lungcondition_20210626.h5')