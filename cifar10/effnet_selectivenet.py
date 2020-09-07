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

class SelectiveNet:
    def __init__(self,dropout=0.2, mc_dropout = 0.2,num_classes=1,
               training=True, input_dim=(224,224,3),pooling="avg"):
        self.c = 0.75
        self.lamda = 32
        self.alpha = 0.5
        self.dropout = dropout
        self.mc_dropout = mc_dropout
        self.pooling = pooling
        self.input_dim = input_dim
        self.training = training
        self.num_classes = num_classes
        
        #create model
        inputs=Input(shape=self.input_dim)
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        base_model.trainable=True
        x = base_model.output
        x = Dropout(self.dropout, name='top_dropout_1')(x,training=self.training)
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)   
        x = Dropout(self.dropout, name='top_dropout_2')(x,training=self.training)
        x = Dense(512,activation='relu', name='dense_512')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.mc_dropout, name='top_dropout_3')(x,training=self.training)
        x = Lambda(lambda x: K.dropout(x, level=self.mc_dropout))(x)

        #classification head (f)
        f = Dense(self.num_classes,activation='softmax', name='f_head')(x)

        #selection head (g)
        g = Dense(512, activation='relu', name='dense_512_g')(x)
        g = BatchNormalization()(g)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        g = Lambda(lambda a: a / 10)(g)
        g = Dense(1, activation='sigmoid',name='g_head')(g)

        # auxiliary head (h)
        selective_output = Concatenate(axis=1, name="selective_head")([f, g])


        auxillary_output = Dense(self.num_classes,activation='softmax', name='auxilary_head')(x)


        self.model = Model(inputs=inputs,outputs=[selective_output,auxillary_output])
        

    def compile_model(self,c=0.75, lamda=32, alpha=0.5,tau=0.5,lr=0.01,decay=1e-6,momentum=0.9):
        self.c = c
        self.lamda = lamda
        self.alpha = alpha
        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], tau), K.floatx())
            return K.mean(g)

        def selective_loss(y_true, y_pred):
            em_coverage = K.mean(y_pred[:, -1])
            loss = K.categorical_crossentropy(K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:,:],
                    y_pred[:, :-1]) + self.lamda * K.maximum(-em_coverage + self.c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], tau), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:,:-1], axis=1), K.argmax(y_pred[:, :-1], axis=1)), 
                             K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())
    
        sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        self.model.compile(loss=[selective_loss, 'categorical_crossentropy'], loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc, coverage])
    
    def model_train(self, train_images, train_labels, EPOCHS, val_ds=None, initial_epoch=0, shuffle=True, callbacks = [], 
                    verbose = 2):
        if(val_ds):
            history =self.model.fit(
                train_images,
                train_labels,
                initial_epoch=initial_epoch,
                epochs=EPOCHS,
                validation_data=val_ds,
                shuffle=shuffle,
                verbose = verbose,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                train_images,
                train_labels,
                initial_epoch=initial_epoch,
                epochs=EPOCHS,
                validation_split=0.2,
                shuffle=shuffle,
                verbose = verbose,
                callbacks=callbacks
            )
        return history         
            
    def selective_predict(self, test):
        gpredictions, _= self.model.predict(test)
        return gpredictions

    def classification_predict(self, test):
        _,predictions= self.model.predict(test)
        return predictions

    def mc_dropout(self,test_ds,iter=100):

        for i in range(iter):
            mc_predictions = self.selective_predict(test_ds)

        mc = np.var(mc_predictions[:,:self.num_classes], 1)
        mc = np.mean(mc,1)
        return -mc

    def selective_risk(self,c,y_true,test_ds):
        y_pred = self.selective_predict(test_ds)
        if(100*(1-c)<0):
            q = 0
        elif(100*(1-c)>100):
            q =100
        else:
            q =100*(1-c)
        threshold = np.percentile(y_pred[:,-1],q)
        covered_indx = y_pred[:, -1] > threshold
        g = covered_indx.astype('int')
        coverage = np.mean(g)
        acc = np.sum( np.equal( np.argmax(y_pred[:, :self.num_classes],axis=1), 
                                 np.argmax(y_true[:, :-1], axis=1))) / np.sum(g)

        risk = 1-acc
        return coverage, risk, acc

    def sr_selective_risk(self,c, y_true, test_ds, mc=False):
        y_pred = self.selective_predict(test_ds)
        if(mc):
            sr = np.max(y_pred[:,:self.num_classes], axis=1)
        else:
            sr = self.mc_dropout(100)
        sr_sorted = np.sort(sr)
        threshold = sr_sorted[sr.shape[0] - int(c * sr.shape[0])]
        covered_indx = sr > threshold
        g = covered_indx.astype('int')
        coverage = np.mean(g)
        acc = np.sum( np.equal( np.argmax(y_pred[:, :self.num_classes],axis=1), 
                                 np.argmax(y_true[:, :-1], axis=1))) / np.sum(g)
        risk = 1-acc
        return coverage, risk, acc
    
    def get_model(self):
        return self.model

    def set_model_weights(self,filename):
        self.model.load_weights(filename)