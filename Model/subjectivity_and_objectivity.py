import os
import re
import numpy as np
import pandas as pd
import sklearn
import init
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
import keras
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten
from keras.layers import Dropout,LSTM,BatchNormalization,LeakyReLU
from keras import metrics
from keras.models import Sequential

from keras.models import load_model
import zipfile

from loadvectors import *
from preprocessing import *
from loadfiles import *
from cnn import *
from lstm import *


def main():
  nltk.download('punkt')
  nltk.download('stopwords')

  stopword = stopwords.words('english')

  path = "" #update your path to folder here

  subjective_file = path+"subjective.txt" #Subjective file
  objective_file = path+"objective.txt"   #Objective file

  #Load files here
  subj,obj = load_files(subjective_file,objective_file)  
  
  #10,000 sentences of which 5000 are subjective and 5000 are objective
  labels = np.zeros(10000)
  labels[0:5000] = 1

  #preprocess the dataset
  corpus = prepare_corpus(stopword)

  #80% for training and 20% for testing

  X_train,X_test,y_train,y_test = train_test_split(corpus,labels,test_size=0.2,random_state=42)

  # Extract contents of zip file
  with zipfile.ZipFile('glove.42B.300d.zip', 'r') as myzip:
    myzip.extractall()

  #This is a function to extract the file from above extracted zip file
  with zipfile.ZipFile('glove.42B.300d.zip.1', 'r') as myzip:
    myzip.extractall()

  #Load the Glove Model
  init.glove_model = load_glove_model()


  train_length = len(X_train)
  test_length = len(X_test)

  #Generate Vectors for training dataset
  train_vectors = generate_vectors(X_train,train_length)

  print(train_vectors[0])

  #Generate vectors for testing data
  test_vectors = generate_vectors(X_test,test_length)

  print(test_vectors[0])

  #Declare the 2D Convolutional Model
  train_cnn_vecs,test_cnn_vecs,model_cnn = cnn_model(train_vectors,test_vectors)
  
  #Train the 2D Convolutional Model
  model_cnn = train_cnn(train_cnn_vecs,test_cnn_vecs,model_cnn)
  
  #Test the 2D Convolutional Model
  accuracy_cnn,f1_cnn = test_cnn(model_cnn,test_cnn_vecs)
  print("CNN model Accuracy is :",accuracy_cnn)
  print("CNN model F1 score is :",f1_cnn)


  #Declare the LSTM model
  train_lstm_vecs,test_lstm_vecs,model_lstm = lstm_model(train_vectors,test_vectors)

  #Train the LSTM Model
  model_lstm = train_lstm(train_lstm_vecs,test_lstm_vecs,model_lstm)


  #Test the LSTM Model
  accuracy_lstm,f1_lstm = test_lstm(model_lstm,test_lstm_vecs)
  print("LSTM model Accuracy is :",accuracy_lstm)
  print("LSTM model F1 score is :",f1_lstm)


if __name__ == '__main__':
  main()