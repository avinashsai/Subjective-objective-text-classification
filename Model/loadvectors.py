import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import init
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_glove_model():
  with open('glove.42B.300d.txt','r',encoding='utf-8') as f:
    init.glove_model = {}
    for line in f:
      splitline = line.split()
      word = splitline[0]
      embedding = np.array([float(val) for val in splitline[1:]])
      init.glove_model[word] = embedding
  
  return init.glove_model



def get_vectors(sentence):
  sents = sentence.split()
  vector = np.zeros(300)
  count = 0
  for word in sents:
    if(word in init.glove_model):
      count+=1
      vector+=init.glove_model[word]
    else:
      vector+=np.random.rand()
  if(count>0):
    vector = (vector/count)
    
  return vector

def generate_vectors(data,data_length):
  data_vectors = np.zeros((data_length,300))
  i = 0
  for sent in data:
    data_vectors[i] = get_vectors(sent)
    i = i+1
  
  return data_vectors
