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


def preprocess(sentence,stopword):
  sentence = re.sub(r"can\'t","can not",str(sentence))
  sentence = re.sub(r"n\'t"," not",str(sentence))
  sentence = re.sub(r"\'ll"," will",str(sentence))
  sentence = re.sub(r"\'s"," is",str(sentence))
  sentence = re.sub(r"\'am"," am",str(sentence))
  sentence = re.sub(r"\'ve"," have",str(sentence))
  sentence = re.sub(r'[^a-zA-Z]'," ",str(sentence))
  new_sents = " "
  sents = word_tokenize(sentence)
  for sent in sents:
    if(sent.lower() not in stopword and len(sent)>1):
      new_sents+=sent.lower()+" "
  return new_sents

def prepare_corpus(stopword):
  corpus = []
  for sent in subj:
    corpus.append(preprocess(sent,stopword))
  for sent in obj:
    corpus.append(preprocess(sent,stopword))
  return corpus

