#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:00:57 2019

@author: sysad
"""

import os
import numpy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

path_dir = '/Users/deepak/Desktop/BTP/codes/lcnn_fft'
#path_dir = '/home/sysad/Desktop/DeepL/codes'
#path_dataset = '/ASVSpoof2017_data_v2/dev'
path_wavfile = '/T_1000001.wav'

os.chdir(path_dir)

fs,x = wavfile.read(path_dir+path_wavfile)
x=x[0:25600-64]
nperseg  = 256
noverlap = 192
nfft     = 512
window= signal.get_window('blackman',nperseg)

f,t,sxx = signal.stft(x,fs,window,noverlap=noverlap,nfft=nfft)
plt.pcolormesh(t,f,abs(sxx))
#plt.imshow(sxx)
#plt.psd(x,fs,NFFT=256,noverlap=noverlap,window = numpy.blackman)
plt.show()