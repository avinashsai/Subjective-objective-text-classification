import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



def load_files(subjective_file,objective_file):
  subj = []
  with open(subjective_file,'r',encoding='latin-1') as f:
    for line in f:
      subj.append(line[:-1])
  obj = []
  with open(objective_file,'r',encoding='latin1') as f:
    for line in f:
      obj.append(line[:-1])
  return subj,obj
