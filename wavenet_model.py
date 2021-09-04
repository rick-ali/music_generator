import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dropout, MaxPool1D, GlobalMaxPool1D, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

def get_wavenet(unique_x, unique_y):
    K.clear_session()
    model = Sequential()
        
    #embedding layer
    model.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 

    model.add(Conv1D(64,3, padding='causal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
        
    model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
            
    #model.add(Conv1D(256,5,activation='relu'))    
    model.add(GlobalMaxPool1D())
        
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(unique_y), activation='softmax'))
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model