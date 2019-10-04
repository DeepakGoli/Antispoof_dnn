#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 04:27:25 2019

@author: sysad
"""

import numpy as np
import keras
from scipy.io import wavfile
from scipy import signal

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates batches of data for a system'
    
    def __init__(self,list_IDs,labels,path_to_dir,batch_size=32,dim=(257,400),
                 n_channels=1,n_classes=2,shuffle=True):
        self.list_IDs  = list_IDs
        self.labels    = labels
        self.batch_size= batch_size
        self.dim       = dim
        self.n_channels= n_channels
        self.n_classes = n_classes
        self.shuffle   = shuffle
        self.path_to_dir = path_to_dir
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        'updates indexes after each epoch. Shuffling of our dataset happens here'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        'Gives number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __data_generation(self,list_IDs_temp):
        'Generates data containing batch_size samples'
        #initialization
        #X=np.empty((self.batch_size,*self.dim,self.n_channels))
        X=np.empty((self.batch_size,*self.dim,self.n_channels))
        y=np.empty((self.batch_size),dtype=int)
        
        #Generate data
        for i,ID in enumerate(list_IDs_temp):
            #store sample
            #X[i,] = np.load('data'+ID+'.npy')
            X[i,] = self.__data_augmentation(ID)
            #sotre class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y,num_classes=self.n_classes)
    
    def __getitem__(self,index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate data
        X,y = self.__data_generation(list_IDs_temp)
        return X,y
    
    def __data_augmentation(self,ID_temp):
        #Generates (256,400,1) dimensional power spectrogram of a speech signal
        fs,x = wavfile.read(self.path_to_dir+ID_temp)
        # 1.6sec @ 16KHz
        duration = 25600-64
        if(len(x)<duration):
            x = np.array(list(x)+list(x[0:duration-len(x)]))
        else :
            x = x[0:duration]
        nperseg = 256
        noverlap = 192
        nfft = 512
        window = signal.get_window('blackman',nperseg)
        f,t,Sxx = signal.stft(x,fs,window,noverlap=noverlap,nfft=nfft)
        log_power = np.empty((*self.dim,self.n_channels))
        log_power[:,:,0] = 20*np.log10(np.power(abs(Sxx),2))
        return log_power
    
        
        
