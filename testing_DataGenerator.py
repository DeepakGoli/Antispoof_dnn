#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:38:34 2019

@author: deepak
"""

import os
import keras
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Conv2D, MaxPooling2D, Maximum
from keras.layers import Cropping2D,Permute,Reshape,Flatten,BatchNormalization
from keras.models import Sequential
from my_classes import DataGenerator
from PIL import Image

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#############################################################################################

#path_to_dir = '/home/sysad/Desktop/DeepL/old/spoof'
path_to_dir = '/Users/deepak/Desktop/BTP/spoofingAttack'
#path_to_trainDataset = '/ASVSpoof2017_data_v2/trainSpectrograms'
#path_to_devDataset   = '/ASVSpoof2017_data_v2/devSpectrograms'
path_to_trainDataset = '/data/train/'
path_to_devDataset   = '/data/dev/'
#path_to_trainProtocol = '/ASVSpoof2017_data_v2/protocol_V2_old/ASVspoof2017_V2_train.trn'
#path_to_devProtocol = '/ASVSpoof2017_data_v2/protocol_V2_old/ASVspoof2017_V2_dev.trl'
path_to_trainProtocol = '/protocol_V2/ASVspoof2017_V2_train.trn.txt'
path_to_devProtocol = '/protocol_V2/ASVspoof2017_V2_dev.trl.txt'

#####################################################################################################

train_protocol = open(path_to_dir+path_to_trainProtocol,"r")
trainProtocols=train_protocol.readlines() 
train_protocol.close()
train_IDs = list(np.zeros(len(trainProtocols))) 
train_labels = list(np.zeros(len(trainProtocols)))
labels={}
for idx,i in enumerate(trainProtocols):
    train_IDs[idx] = i.split()[0]  
    label = i.split()[1]
     # spoof=0 , genuine=1
    train_labels[idx] = 0 if label == 'spoof' else 1
    labels[train_IDs[idx]]=train_labels[idx]

dev_protocol = open(path_to_dir+path_to_devProtocol)
devProtocols=dev_protocol.readlines()
dev_protocol.close()
dev_IDs = list(np.zeros(len(devProtocols)))
dev_labels = list(np.zeros(len(devProtocols)))
for idx,i in enumerate(devProtocols):
    dev_IDs[idx] = i.split()[0]
    label = i.split()[1]
    # spoof=0 , genuine=1
    dev_labels[idx] = 0 if label == 'spoof' else 1
    labels[dev_IDs[idx]]=dev_labels[idx]

params={'batch_size':32,'n_classes':2,'shuffle':True}
partition = {'train':train_IDs,'validation':dev_IDs[:150]+dev_IDs[len(dev_IDs)-150:]}

training_generator = DataGenerator(partition['train'], labels, **params,path_to_dir=path_to_dir+path_to_trainDataset)
validation_generator = DataGenerator(partition['validation'],labels,**params,path_to_dir=path_to_dir+path_to_devDataset)

############################################################################################

input_shape=(257,400,1)
num_classes=2

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

input_image = Input(input_shape)
out = lightCNN(inputs = input_image)

model = Model(inputs=[input_image], outputs = out)
    
#####################################################################################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=training_generator,epochs=10,validation_data=validation_generator,
                    use_multiprocessing=True,workers=4)
#model.fit_generator(dev_generator,epochs = 10,steps_per_epoch=57,validation_data=validation_generator,validation_steps=10)
model.save('testing_DataGenerator.h5')