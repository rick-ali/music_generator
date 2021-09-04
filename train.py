import os
from os.path import join
import numpy as np
from utilities import read_midi
from collections import Counter
from sklearn.model_selection import train_test_split
from wavenet_model import get_wavenet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import scipy.io as sio

data_path = './processed_data/beeth_processed.mat'

data = {}
sio.loadmat(data_path, mdict=data)
x_tr = data["x_tr"]
y_tr = data["y_tr"][0]
x_val = data["x_val"]
y_val = data["y_val"][0]
unique_x = data["unique_x"]
unique_y = data["unique_y"]

del data

model = get_wavenet(unique_x, unique_y)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history = model.fit(x_tr, y_tr, batch_size=128, epochs=50, validation_data=(x_val, y_val), verbose=1, callbacks=[mc])
