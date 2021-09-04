import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utilities import convert_to_midi
import scipy.io as sio

def inference(x_val, no_of_timesteps, unique_x):
    model = load_model('best_model.h5')

    ind = np.random.randint(0,len(x_val)-1)

    random_music = x_val[ind]
    print(random_music)

    random_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    val_notes = [random_int_to_note[i] for i in random_music]
    convert_to_midi(val_notes, out_name='original.mid')

    predictions=[]
    for i in range(15):

        random_music = random_music.reshape(1,no_of_timesteps)

        prob  = model.predict(random_music)[0]
        y_pred = np.argmax(prob,axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
        random_music = random_music[1:]
        
    print(predictions)

    # Convert integers back into notes
    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
    predicted_notes = [x_int_to_note[i] for i in predictions]
    
    convert_to_midi(predicted_notes)

if __name__ == "__main__":
    no_of_timesteps = 32
    
    data_path = './processed_data/beeth_processed.mat'
    data = {}
    sio.loadmat(data_path, mdict=data)

    inference(data["x_val"], no_of_timesteps, data["unique_x"])