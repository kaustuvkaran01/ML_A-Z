# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:06:04 2020

@author: Asus
"""
#Convolutional Neural Network
##Installing Theano
#pip isntall --upgrade --no-deps git+git://gtihub.com/Theano/Theano.git 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Init the CNN
classifier=Sequential()

#Adding the First Layer of the CNN

#Step1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step2 Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))
