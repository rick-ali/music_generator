import os
from os.path import join
import numpy as np
from utilities import read_midi
from collections import Counter
from sklearn.model_selection import train_test_split
import scipy.io as sio

data_dir = './music/beeth'

files = [f for f in os.listdir(data_dir) if f.endswith(".mid")]

notes_array = np.array([read_midi(join(data_dir,f)) for f in files], dtype="object")

# converting 2D array into 1D array
notes_ = [element for note_ in notes_array for element in note_]  #? Can't just np.flatten()?

# prepare new musical files which contain only the top frequent notes
freq = dict(Counter(notes_))
frequent_notes = [note_ for note_, count in freq.items() if count>=50]

new_music=[]

for notes in notes_array:
    temp=[]
    for note_ in notes:
        if note_ in frequent_notes:
            temp.append(note_)            
    new_music.append(temp)
    
new_music = np.array(new_music, dtype="object")


# Prepare the data
no_of_timesteps = 32
x = []
y = []

for note_ in new_music:
    for i in range(0, len(note_) - no_of_timesteps, 1):
        #preparing input and output sequences
        input_ = note_[i:i + no_of_timesteps]
        output = note_[i + no_of_timesteps]
        
        x.append(input_)
        y.append(output)
        
x=np.array(x)
y=np.array(y)

# Assign a unique integer to every note
unique_x = list(set(x.ravel()))
x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

#preparing input sequences
x_seq=[]
for i in x:
    temp=[]
    for j in i:
        #assigning unique integer to every note
        temp.append(x_note_to_int[j])
    x_seq.append(temp)
    
x_seq = np.array(x_seq)


unique_y = list(set(y))
y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y)) 
y_seq=np.array([y_note_to_int[i] for i in y])

# TRAIN TEST SPLIT
x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

sio.savemat(join("processed_data", "beeth_processed.mat"), {'x_tr': x_tr, 'x_val': x_val, 'y_tr': y_tr, 'y_val': y_val, 'unique_x': unique_x, 'unique_y': unique_y})

print("Data saved at " + str(join("processed_data", "beeth_processed.mat")))
