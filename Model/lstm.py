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

def lstm_model(train_lstm_vecs,test_lstm_vecs):
  train_lstm_vecs = train_lstm_vecs.reshape((train_length,300,1))
  test_lstm_vecs = test_lstm_vecs.reshape((test_length,300,1))

  model_lstm = Sequential()
  model_lstm.add(LSTM(32,return_sequences=True,input_shape=(300,1)))
  model_lstm.add(LSTM(64))
  model_lstm.add(Dense(1,activation="sigmoid"))
  
  return train_lstm_vecs,test_lstm_vecs,model_lstm

def train_lstm(train_vecs,test_vecs,model_lstm):
  model_lstm.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
  model_lstm.fit(train_vecs,y_train,batch_size=64,epochs=75,validation_data=(test_vecs,y_test))
  model_lstm.save('glove_lstm.h5')
  return model_lstm

def test_lstm(model_lstm,test_vecs):
  pred = model_lstm.predict(test_vecs)
  y_pred = np.zeros(test_length)
  for i in range(test_length):
    if(pred[i]>=0.5):
      y_pred[i]=1
  
  accuracy = accuracy_score(y_test,y_pred)
  f1 = f1_score(y_test,y_pred)
  return accuracy,f1

