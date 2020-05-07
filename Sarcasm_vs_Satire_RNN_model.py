# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:31:37 2020

@author: abhi0
"""

#Sarcasm data-set:

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split 
from keras.regularizers import l2
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import ModelCheckpoint
import json

def parse_data(file):

    for l in open(file,'r'):
        yield json.loads(l)
    
data =\
list(parse_data('C:/Users/abhi0/OneDrive/Documents/Data_on_Sattire_and_Sarcasm/Sarcasm_Headlines_Dataset.json'))    

#Not used to avoid severe class-imbalance. Model would never generalise to
#any sort of data in the world, if used, without increasing the sarcasm
#data-set. Not using this gives a good ratio of the two classes. 
data2=\
list(parse_data('C:/Users/abhi0/OneDrive/Documents/Data_on_Sattire_and_Sarcasm/Sarcasm_Headlines_Dataset_v2.json'))    

#Extracting the labels,indices and the headlines corresponding to sarcastic
#statements for the first set
sarIdx=[]
Headline=[]
Labels=[]

for i in range(len(data)):
    if data[i]['is_sarcastic']==1:
        sarIdx.append(i)
        Headline.append(data[i]['headline'])
        Labels.append('Sarcasm')       
        
        
###'sattire' data-set: 
import pandas as pd
data3=pd.read_csv('C:/Users/abhi0/OneDrive/Documents/Data_on_Sattire_and_Sarcasm/OnionOrNot.csv')        

satIdx=[]

#Sarcasm ending Index:
sarEndIdx=len(Headline)        

for j in range(len(data3)):
    if data3['label'][j]==1:
        satIdx.append(j)
        Labels.append('Satire') 
        Headline.append(data3['text'][j])
        
        
#merging columsn into a dataframe:
df = pd.DataFrame(
    {'Headlines': Headline,
     'Labels': Labels,
    })        

    
import numpy as np
from sklearn.model_selection import train_test_split
# The maximum number of words to be used. (most frequent)
from keras.preprocessing.text import Tokenizer
MAX_NB_WORDS = 50000
# Max number of words in each headline.
MAX_SEQUENCE_LENGTH = 300
# This is fixed.
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Headlines'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df['Headlines'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
Y= np.array(df['Labels'])
from keras.utils import to_categorical
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y=le.fit_transform(Y)
print(Y.shape)
#Y= to_categorical(Y)
#print(Y.shape)
#print(df['Labels'][1])
print(Y[len(Y)-1])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42,shuffle=True)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

from keras.layers import SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau

def lr_scheduler(epoch):
    
    learningVect=[0.001,0.005,0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.5]
    lr =learningVect[epoch]       
    
    return lr

#keras.callbacks.callbacks.BaseLogger(stateful_metrics=None)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
#model.add(SpatialDropout1D(0.000000000002))
model.add(LSTM(200,return_sequences=True,recurrent_dropout=0.5))
model.add(LSTM(200,return_sequences=True,recurrent_dropout=0.5))
model.add(LSTM(200,return_sequences=True,recurrent_dropout=0.5))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

#history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.3,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

learnPlatReducer=\
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,\
                                 verbose=1, mode='auto', min_delta=0.0001,\
                                 cooldown=0, min_lr=0)
mdl_chk=\
ModelCheckpoint('BestModel.h5',monitor='val_loss', verbose=1, save_best_only=True,\
                save_weights_only=True)

history =\
model.fit(X_train, Y_train, epochs=10, batch_size=30,\
          validation_split=0.3,\
          callbacks= [learnPlatReducer,mdl_chk])
    
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()    