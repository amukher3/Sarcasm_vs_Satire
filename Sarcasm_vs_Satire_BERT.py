# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:01:17 2020

@author: abhi0
"""


##############################################################################
############# Loading the libraries and files #############################
#############################################################################
from numpy import array
import json
from fastai.text import *
import numpy as np
from sklearn.feature_selection import chi2
import ktrain
from ktrain import text




#Loading the files:
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
    {'Labels': Labels,
     'Headlines': Headline,
    })        
    
    
tempX=np.array(df['Headlines'])
tempY=np.array(df['Labels'])


#############################################################################
########### Separating into train-Dev and preprocessing #####################
#############################################################################

#Creating training and validation splits:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(tempX,tempY, test_size = 0.3,shuffle=True)



#############################################################################
############# Defining the transformer-model ###############################
############################################################################

#MODEL_NAME='distilbert-base-uncased'
## Using BERT for now
MODEL_NAME='bert-base-uncased'
#MODEL_NAME='bert'


t=text.Transformer(MODEL_NAME)

trn=t.preprocess_train(X_train,y_train)
val=t.preprocess_test(X_test,y_test)

model=t.get_classifier()
learner=ktrain.get_learner(model,train_data=trn,val_data=val,batch_size=6)

learner_found=learner.lr_find(show_plot=True,max_epochs=4)

learner.fit_onecycle(5e-5,40)

predictor=ktrain.get_predictor(learner.model,preproc=t)

#Saving the weights:
predictor.save('C:/Users/abhi0/OneDrive/Documents/separating_sarcasm_from_satire')

#Doing prediction:
predictor.predict('Due to the rising cost of ammunition I am no longer able to afford a warning shot.')

#Getting the probabilities:
predictor.predict_proba('Due to the rising cost of ammunition I am no longer able to afford a warning shot.')