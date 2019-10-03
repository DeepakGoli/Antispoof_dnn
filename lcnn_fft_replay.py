#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 04:02:40 2019

@author: sysad
"""

import os
import keras
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Conv2D, MaxPooling2D, Maximum
from keras.layers import Cropping2D,Permute,Reshape,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

path_dir = '/home/sysad/Desktop/DeepL/old/spoof'
path_train_dataset = '/ASVSpoof2017_data_v2/trainSpectrograms'
path_dev_dataset   = '/ASVSpoof2017_data_v2/devSpectrograms'

#####################################################################################################

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer='he_normal', name=name)(x)
  if not use_bias:
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  return x

def mfm(x):
    shape = K.int_shape(x)
    x  = Permute(dims=(3,2,1))(x) #swapping 1st and 3rd axis
    x1 = Cropping2D(cropping=((0,shape[3]//2),0))(x) 
    x2 = Cropping2D(cropping=((shape[3]//2,0),0))(x)
    x = Maximum()([x1 , x2])
    x = Permute(dims=(3,2,1))(x) #swapping 1st and 3rd axis
    x = Reshape([shape[1],shape[2],shape[3]//2])(x)
    return x
def Conv2dMFM(x,f1=32,k1=(1,1),s1=(1,1),f2=32,k2=(1,1),s2=(1,1),k3=(2,2),s3=(2,2)):
    net_temp = Conv2D(f1,kernel_size=k1,strides=s1)(x)
    net_temp = mfm(net_temp)
    net_temp = Conv2D(f2,kernel_size=k2,strides=s2)(net_temp)
    net_temp = mfm(net_temp)
    net_temp  = MaxPooling2D(pool_size=k3,strides=s3)(net_temp)
    return net_temp
def lightCNN(inputs):
    net = Conv2D(32,kernel_size=(5,5),strides=(1,1),input_shape=input_shape)(inputs)
    print("executed")
    net = mfm(net)
    net1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(net)
    #net2 = Conv2D(32,kernel_size=(1,1),strides=(1,1))(net1)
    #net2 = mfm(net2)
    net2 = Conv2dMFM(net1,f1=32,k1=(1,1),s1=(1,1),f2=48,k2=(3,3),s2=(1,1),k3=(2,2),s3=(2,2))
    net3 = Conv2dMFM(net2,f1=48,k1=(1,1),s1=(1,1),f2=64,k2=(3,3),s2=(1,1),k3=(2,2),s3=(2,2))
    net4 = Conv2dMFM(net3,f1=64,k1=(1,1),s1=(1,1),f2=32,k2=(3,3),s2=(1,1),k3=(2,2),s3=(2,2))
    net5 = Conv2dMFM(net4,f1=32,k1=(1,1),s1=(1,1),f2=32,k2=(3,3),s2=(1,1),k3=(2,2),s3=(2,2))
    net5 = Flatten()(net5)
    net6_1 = Dense(32)(net5)
    net6_2 = Dense(32)(net5)
    net6 = Maximum()([net6_1 , net6_2])
    net7 = Dense(num_classes,activation = 'softmax')(net6)
    return net7

#####################################################################################################

input_image = Input(shape = input_shape)
out = lightCNN(inputs = input_image)

model = Model(inputs=[input_image], outputs = out)
    
#####################################################################################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_dataGenerator,epochs = 20,steps_per_epoch=60,validation_data=validation_generator,validation_steps=20)
#model.fit_generator(dev_generator,epochs = 10,steps_per_epoch=57,validation_data=validation_generator,validation_steps=10)
model.save('lcnn_fft.h5')
