import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

import tensorflow as tf
import keras
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten
from keras.layers import Dropout,LSTM,BatchNormalization,LeakyReLU
from keras import metrics
from keras.models import Sequential

from keras.models import load_model

def cnn_model(train_vecs,test_vecs):
  train_cnn_vecs = train_vecs.reshape((train_length,20,15,1))
  test_cnn_vecs = test_vecs.reshape((test_length,20,15,1))
  model_cnn = Sequential()
  model_cnn.add(Convolution2D(32,kernel_size=(3,3),input_shape=(20,15,1)))
  model_cnn.add(LeakyReLU())
  model_cnn.add(BatchNormalization())
  model_cnn.add(MaxPooling2D((2,2),padding="same"))
  model_cnn.add(Convolution2D(64,kernel_size=(3,3)))
  model_cnn.add(LeakyReLU())
  model_cnn.add(BatchNormalization())
  model_cnn.add(MaxPooling2D((2,2),padding="same"))
  model_cnn.add(Convolution2D(128,kernel_size=(3,3)))
  model_cnn.add(LeakyReLU())
  model_cnn.add(BatchNormalization())
  model_cnn.add(MaxPooling2D((2,2),padding="same"))
  model_cnn.add(Flatten())
  model_cnn.add(Dense(128))
  model_cnn.add(LeakyReLU())
  model_cnn.add(Dense(32))
  model_cnn.add(LeakyReLU())
  model_cnn.add(Dense(8))
  model_cnn.add(LeakyReLU())
  model_cnn.add(Dense(1,activation="sigmoid"))
  
  return train_cnn_vecs,test_cnn_vecs,model_cnn

def train_cnn(train_vecs,test_vecs,model_cnn):
  model_cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
  model_cnn.fit(train_vecs,y_train,batch_size=64,epochs=50,validation_data=(test_vecs,y_test))
  model_cnn.save('glove_cnn2d.h5')
  return model_cnn

def test_cnn(model_cnn,test_vecs):
  pred = model_cnn.predict(test_vecs)
  y_pred = np.zeros(test_length)
  for i in range(test_length):
    if(pred[i]>=0.5):
      y_pred[i]=1
  
  accuracy = accuracy_score(y_test,y_pred)
  f1 = f1_score(y_test,y_pred)
  return accuracy,f1
