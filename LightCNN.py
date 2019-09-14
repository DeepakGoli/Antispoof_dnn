#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:22:46 2019

@author: Deepak kanna
"""

import os
import keras
import keras.backend as K
import h5py
from keras.models import Model
from keras.layers import Input,Dense,Conv2D, MaxPooling2D, Maximum
from keras.layers import Cropping2D,Permute,Reshape,Flatten
from keras.datasets import mnist

batch_size = 128
num_classes = 10

img_rows, img_cols = 28, 28

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test  = x_train.reshape(x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

#####################################################################################################

def mfm(x):
    shape = K.int_shape(x)
    x  = Permute(dims=(3,2,1))(x) #swapping 1st and 3rd axis
    x1 = Cropping2D((0,shape[3]//2),0)(x) 
    x2 = Cropping2D((shape[3]//2,0),0)(x)
    x = Maximum()([x1 , x2])
    x = Permute(dims=(3,2,1))(x) #swapping 1st and 3rd axis
    x = Reshape([shape(1),shape(2),shape(3)//2])(x)
    return x
def Conv2dMFM(x,f1,k1=(1,1),s1=(1,1),f2,k2=(1,1),s2=(1,1),k3,s3):
    net_temp = Conv2D(f1,kernel_size=k1,strides=s1)(x)
    net_temp = mfm(net_temp)
    net_temp = Conv2D(f2,kernel_size=k2,strides=s2)(net_temp)
    net_temp = mfm(net_temp)
    net_temp  = MaxPooling2D(pool_size=k3,stride=s3)
    return net_temp
def lightCNN(inputs):
    net = Conv2D(32,kernel_size=(5,5),strides=(1,1),input_shape=input_shape)(inputs)
    net = mfm(net)
    net1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))
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
model.compile()
    
    
    
    
    
    