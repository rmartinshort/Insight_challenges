#!/usr/bin/env python
#RMS 2019
#Functions to aid in data cancer detection data challenge solution

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, \
precision_score, roc_auc_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_validate

from imblearn.under_sampling import RandomUnderSampler

def precision_recall(Y_true,Y_scores):
    
    '''Plot a PR curve as a function of threshold'''
    
    precision, recall, thresholds = precision_recall_curve(Y_true, Y_scores)
    
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.show()
    

def plot_2d_space(X, y, label='Classes'): 
    
    '''Do a PCA and plot 2D space showing the classes'''
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
def remove_features_one_by_one(X,y):
    
    '''
    Split X and y into train and test, fit model, 
    find feature importances, remove features one by one by
    dropping the feature with the lowest importance. 
    
    returns lists containing scores for each number of features and the 
    features to preserve
    '''
    
    feature_names = list(X.columns)

    #Generate holdout set
    X_tt, X_holdout, y_tt, y_holdout = train_test_split(X, y, test_size=0.3)
    
    #Do random under sampling (shown to improve the recall score)
    rus = RandomUnderSampler(return_indices=False)
    
    #These datasets will be used for test/train 
    X_rus, y_rus = rus.fit_sample(X_tt, y_tt)
    
    #We're going to set up the model several times, dropping one feature each time
    cols_to_use = feature_names.copy()
    
    holdout_recall_scores = []
    holdout_precision_scores = []
    holdout_f1_scores = []
    used_columns = {}
    
    for i in range(len(feature_names),0,-1):
        
        print('\n-----------------------------------')
        print('Number of features: %g' %i)
        
        used_columns[i] = cols_to_use.copy()
        
        X_use = X_tt[cols_to_use.copy()]
        X_holdout_use = X_holdout[cols_to_use.copy()]
        
        y_use = y_tt
        y_holdout_use = y_holdout
        
        #These datasets will be used for test/train 
        X_rus, y_rus = rus.fit_sample(X_use, y_use)
        
        LR = LogisticRegression(penalty='l1',solver='liblinear')
        scores = cross_validate(LR,X_rus,y_rus,cv=5,scoring='f1')
        
        print('Testing scores from cross validation')
        print(scores['test_score'])
        print('-----------------------------------')
        
        #Fit to the full test/train dataset in order to predict holdout
        LR.fit(X_rus,y_rus)
        
        #predict holdout set
        y_class = LR.predict(X_holdout_use)
        
        #holdout scores
        score_holdout_f1 = f1_score(y_holdout_use,y_class)
        score_holdout_recall = recall_score(y_holdout_use,y_class)
        score_holdout_precision = precision_score(y_holdout_use,y_class)
        
        print('F1 holdout score %g on %g features' %(score_holdout_f1,len(cols_to_use)))
        
        print('-----------------------------------')
        
        holdout_recall_scores.append(score_holdout_recall)
        holdout_precision_scores.append(score_holdout_precision)
        holdout_f1_scores.append(score_holdout_f1)
        
        #Put the feature importances in order
        fimportances = LR.coef_[0]
        feature_order = np.argsort(abs(fimportances))
        
        for i in range(len(cols_to_use)-1,-1,-1):
            ind = feature_order[i]
            print('Feature: %s, importance: %.3f' %(cols_to_use[ind],abs(fimportances[ind])))
        
        #Drop the least important feature from the columns 
        cols_to_use.pop(feature_order[0])
        
        print('-----------------------------------')
        
    return holdout_recall_scores, holdout_precision_scores, holdout_f1_scores, used_columns