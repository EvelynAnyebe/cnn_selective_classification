# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers,losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Concatenate,Lambda,Activation, Input
from tensorflow.keras.models import Model
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import backend as K

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def SR_model(num_classes, dropout,mc_dropout, input_dim, training, pooling='avg'):
    inputs=Input(input_dim)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable=True
    x = base_model.output
    x = Dropout(dropout, name='top_dropout_1')(x,training=training)
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)   
    x = Dropout(dropout, name='top_dropout_2')(x,training=training)
    x = Dense(512,activation='relu', name='dense_512')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout, name='top_dropout_3')(x,training=training)
    x = Lambda(lambda x: K.dropout(x, level=mc_dropout))(x)

    #classification head (f)
    sr = Dense(num_classes,activation='softmax', name='dense_f')(x) 
    return Model(inputs=inputs,outputs=sr)