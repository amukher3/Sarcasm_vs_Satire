# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:26:16 2020

@author: abhi0
"""
#Sarcasm data-set:
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
        Labels.append(1)       
        
        
###'sattire' data-set: 

import pandas as pd

data3=pd.read_csv('C:/Users/abhi0/OneDrive/Documents/Data_on_Sattire_and_Sarcasm/OnionOrNot.csv')        

satIdx=[]

#Sarcasm ending Index:
sarEndIdx=len(Headline)        

for j in range(len(data3)):
    if data3['label'][j]==1:
        satIdx.append(j)
        Labels.append(2) #Satire is labeled as '2'
        Headline.append(data3['text'][j])
        
        
#merging columsn into a dataframe:
df = pd.DataFrame(
    {'Headlines': Headline,
     'Labels': Labels,
    })        

    
import numpy as np
from sklearn.model_selection import train_test_split

#Depedndent columns: 
X=df['Headlines']
y=df['Labels']

predSetX=[]
predSetY=[]

#Train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,\
                                                random_state=42,shuffle=True)

cutIdx=0.3

#Separate a pred set:
predSetX=X_train[:round(len(X)*cutIdx)]
predSetY=y_train[:round(len(y)*cutIdx)]

#Removing the prediction set 
X_train=X_train[len(predSetX):]
y_train=y_train[len(predSetY):]

def gettingAUC(scores,Labels):
    
    fpr, tpr, thresholds=\
    metrics.roc_curve(Labels,scores[:,1], pos_label=2)
    aucVal=metrics.auc(fpr, tpr)
    
    return aucVal
    
    


################################## Model ######################################

#Although for the considered data-set, the headlines can be easily separated by length.
#length is not considered here as an feature.Since in general satire and sarcasm
#have nothing to do with the length of the statement/headline.Using length of the
#headline could overfit the model for this data-set thereby making it a poor and less
#robust model in general. Something which will break as soon as the property 
#gets dissatisfied. There can be ample example where satire and sarcastic statements
#have the same length.  

#Satire and Sarcasm should be a function of the sentence and should be only 
#dependent on the words and their interaction in the sentence. Very similar to
#a positive sentence vs. a negative sentence, sentence length should not be 
#a feature to be used for classification. 

#P.S:The samples considered here are perfect example of biased sampling since the
#samples in one of the classes are clearly biased as far as the length of the 
#individual examples are considered. 

###############################################################################

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
from xgboost import XGBClassifier

accScore_train=[]
accScore_test=[]
n_estimatorsIdx=[]
max_depthIdx=[]
min_leaf_samplesIdx=[]
min_split_samplesIdx=[]
bootstrapIdx=[]
Indices=[]
aucVal_train=[]
aucVal_test=[]
recallVal_test=[]
precisionVal_test=[]


def Model(X_train, X_test, y_train, y_test):
    
    #number of CPU cores:
    cores=multiprocessing.cpu_count()-2
    
    #parameter space to be searched over:
    # A parameter grid for XGBoost:
    parameters = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'learning_rate':[0.01,0.04,0.08,0.1,0.3,0.5,0.8],
            'n_estimators':[100,300,500,700,900]
                }
                  
                      
    #different models
    model=XGBClassifier(objective='binary:logistic')
    
    #Grid search model:
    clf= GridSearchCV(model, param_grid=parameters, cv=5,n_jobs=4)
    
    #Differnt models in the pipeline:
    text_clf=\
    Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),\
              ('xgBoost', clf)])
            
    ############################################        
    ########## For the training-set ############
    ############################################
                            
    #fitting the grid to the train data
    text_clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    
    #Predicting training set-accuracy:
    predictions_train=text_clf.predict(X_train)
    accScore_train.\
    append(metrics.accuracy_score\
           (y_train,predictions_train))
                            
    #Getting AUC's for the train-set:
    scores=[]
    Labels=[]
    scores=text_clf.predict_proba(X_train)
    Labels=y_train
    aucVal_train.append(gettingAUC(scores,Labels))
                            
    ###########################################                        
    ########## For the test-set ###############
    ###########################################
                        
    #Form a test set:
    predictions_test=text_clf.predict(X_test)
                            
    #Report the confusion matrix:   
    print(metrics.confusion_matrix(y_test,predictions_test))
                        
    #Print a classification report:
    print(metrics.classification_report(y_test,predictions_test))   
    
    #test-set accuracy:
    accScore_test.append(metrics.accuracy_score(y_test,predictions_test))
                            
    #Getting AUC's for the test-set:
    scores=[]
    Labels=[]
    scores=text_clf.predict_proba(X_test)
    Labels=y_test
    aucVal_test.append(gettingAUC(scores,Labels))
                            
#    #Extracting the recall and Precision value for the
#    #test set for the positive class(class-2/satire):
#    totReport=metrics.accuracy_score(y_test,predictions_test)
#    recallVal_test.append(totReport[138:142])
#    precisionVal_test.append(totReport[128:132])
                                                                          
    return accScore_train,accScore_test,aucVal_test,\
            aucVal_train,recallVal_test,precisionVal_test
    

accScore_train,accScore_test,aucVal_test,\
aucVal_train,recallVal_test,precisionVal_test=\
                                    Model(X_train, X_test, y_train, y_test)

#Print the overall accuracy
print(f"AUC for the training set is:{aucVal_train}")
print(f"AUC for the test set is:{aucVal_test}")

###############################################################################
##################### For the prediction-set ##################################
###############################################################################

##verifying for the train-set and the test-set
##bestIdx=np.argmax(abs(np.subtract(aucScore_train,aucScore_test)))
#bestIdx=np.argmax(aucScore_test)
#
##Separating into train-test: 
#
##Separate-ped set:
#predSetX_test=predSetX[:round(len(predSetX)*0.30)]
#predSetY_test=predSetY[:round(len(predSetY)*0.30)]
#
##Removing the prediction set 
#predSetX_train=predSetX[len(predSetX_test):]
#predSetY_train=predSetY[len(predSetY_test):]
#
#
##Random-forest model on the left out set
#model=RandomForestClassifier(n_estimators=n_estimatorsIdx[bestIdx],\
#                             max_features='auto',\
#                             max_depth=max_depthIdx[bestIdx],\
#                             min_samples_split=min_split_samplesIdx[bestIdx],\
#                             min_samples_leaf=min_leaf_samplesIdx[bestIdx],\
#                             bootstrap=bootstrapIdx[bestIdx])
##Pipelinng the model:
#text_clf= Pipeline([('tfidf', TfidfVectorizer()),('RandomForest', model)])
#
#
##fitting the grid to the train data or any data for that sake 
##as long as they are from the same whole data-set...
#text_clf.fit(predSetX_train,predSetY_train)
#
##Predicting pred-set-accuracy:
#scores=text_clf.predict_proba(predSetX_test)
#Labels=predSetY_test
#aucVal=gettingAUC(scores,Labels)
#print(f"AUC on the left out set is {aucVal}")

