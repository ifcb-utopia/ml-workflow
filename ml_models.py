# library of functions to define machine learning models
# some functions created by Emmett Culhane, copied from cnn-12V1.ipynb and model_fuction_library.py,
# and with docstrings and some changes added here


import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
import os, re
import cv2
import locale

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json


def create_cnn_model_A1(width, height, depth, filters=(32, 16, 64, 32, 128, 128, 64, 256, 256, 128), regress=False):
    inputShape = (height, width, depth)
    chanDim = -1
    inputs = Input(shape=inputShape)
    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        if i in [1, 3, 6, 9]:
            x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(1000)(x)
    x = Activation("relu")(x)
    if regress:
        x = Dense(1, activation="linear")(x)
    model = Model(inputs, x)
    return model