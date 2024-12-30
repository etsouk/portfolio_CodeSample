# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:12:43 2024

@author: Vagelis Tsoukas
"""

#import A_import_libraries

import B_data_cleaning_normalized

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, roc_auc_score, make_scorer, recall_score, RocCurveDisplay, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate

from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from collections import Counter
from itertools import combinations

import scipy as sc
from scipy import stats

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting library

from scipy.stats import zscore





def classifierChoice(X_train, X_test, y_train, y_test, clChoice, datafr_without_class, num=20):

    if (clChoice == "RF"):
        
        clf = RandomForestClassifier()
            
        clf.fit(X_train, y_train)
        feature_importance = clf.feature_importances_
        
        df_feat_import = pd.DataFrame(feature_importance, columns=["importance"])
        df_feat_import.index = datafr_without_class.columns
        df_feat_import = df_feat_import.sort_values('importance', ascending=False)
        df_feat_import = df_feat_import.index[:num].to_list()

        
    elif (clChoice == "XG"):
        
        clf = XGBClassifier()
            
        clf.fit(X_train, y_train)
        feature_importance = clf.feature_importances_
            
        df_feat_import = pd.DataFrame(feature_importance, columns=["importance"])
        df_feat_import.index = datafr_without_class.columns
        df_feat_import = df_feat_import.sort_values('importance', ascending=False)
        df_feat_import = df_feat_import.index[:num].to_list()
    
    elif (clChoice == "EX_T"):
        clf = ExtraTreesClassifier()
        
        clf.fit(X_train, y_train)
        feature_importance = clf.feature_importances_
            
        df_feat_import = pd.DataFrame(feature_importance, columns=["importance"])
        df_feat_import.index = datafr_without_class.columns
        df_feat_import = df_feat_import.sort_values('importance', ascending=False)
        df_feat_import = df_feat_import.index[:num].to_list()

    elif (clChoice == "ADA"):
       
        clf = AdaBoostClassifier()
        
        clf.fit(X_train, y_train)
        feature_importance = clf.feature_importances_
            
        df_feat_import = pd.DataFrame(feature_importance, columns=["importance"])
        df_feat_import.index = datafr_without_class.columns
        df_feat_import = df_feat_import.sort_values('importance', ascending=False)
        df_feat_import = df_feat_import.index[:num].to_list()

    elif (clChoice == "GB"):
        
        clf = GradientBoostingClassifier()
        
        clf.fit(X_train, y_train)
        feature_importance = clf.feature_importances_
            
        df_feat_import = pd.DataFrame(feature_importance, columns=["importance"])
        df_feat_import.index = datafr_without_class.columns
        df_feat_import = df_feat_import.sort_values('importance', ascending=False)
        df_feat_import = df_feat_import.index[:num].to_list()
    
    return (df_feat_import)        





def HoldOut_sign_feat(df_without_class, datafr_class_column, clChoice):
    """
    ## This function returns a list that contains 200 lists. Each one of those lists contains 
    ## the most 20 important features of the classifier during an epoch. On each epoch, the data frame
    ## is subjected to independent trai_test_split.
    ## A total of 200 epochs are called for statistical purposes.
    """
    
    epochs=200
    
    signif_feat_list = []
    
    for i in range(epochs):
        print(f"the epoch of significance is: {i}")
        (X_train, X_test, y_train, y_test) = train_test_split(df_without_class, 
                                                              datafr_class_column, 
                                                              test_size=0.3, 
                                                              stratify=datafr_class_column)
   
        (signif_feats)= classifierChoice(X_train, 
                                              X_test, 
                                              y_train, 
                                              y_test,
                                              clChoice,
                                              df_without_class) 
                                              
        signif_feat_list.append(signif_feats)
        
    return(signif_feat_list)


def most_frequent_feat(list_of_feat_lists, num=20):
    """
    ## This function returns the features that are the 20 most frequent ones on the 
    ## 200-epochs list and the respective counts of them.
    ## The counts are returned, in case we want to check the frequency of features on 200-epochs list.
    
    Generally, it refers to the frequency of elements in a list of lists.
    """

    all_feat_together = []
    for sublist in list_of_feat_lists:
        for feat in sublist:
            all_feat_together.append(feat)
    #all_feat_together = [feat for sublist in list_of_feat_lists for sub in sublist for feat in sub] 
    ## alternative code for the loop
        
    feat_counts = Counter(all_feat_together)
    most_common_feat = feat_counts.most_common(num)
    
    counts = []
    signif_features = []
    for feat, count in most_common_feat:
        signif_features.append(feat)
        counts.append(count)
    #signif_features = [feat for feat, count in signif_features]
    ## alternative code for the loop
    
    return(signif_features, counts)   




####---- Main functions ---- #####

def import_feat_per_classifier(datafr_scaled, clChoice="XG"):
    #clChoice = "RF" "XG" "EX_T" "ADA" "GB"   
    ## potential classifiers for feature importance extraction.
    ## In case of the addition of another classifier, we have to add the respective code on
    ## "classifierChoice" fuction and the respective abbrevation on this function. 
    """
    This function takes a scaled data frame and returns a scaled data frame with columns
    only the 20 most important features according to a specific classifier (plus the "class" column)
    """
    
    ## The below line of code returns a list that contains 200 lists. Each one of those lists contains 
    ## the most 20 important features of the classifier during an epoch. On each epoch, the data frame
    ## is subjected to independent trai_test_split.
    ## A total of 200 epochs are called for statistical purposes.
    signif_feat_list = HoldOut_sign_feat(datafr_scaled.iloc[:, :-1], datafr_scaled.iloc[:, -1], clChoice)
    
    ## The below line of code returns the features that are the 20 most frequent ones on the 
    ## 200-epochs list and the respective counts of them.
    ## The counts in case we want to check the frequency of features on 200-epochs list.
    signif_feat, signif_counts = most_frequent_feat(signif_feat_list)
    
    datafr_scaled_with_import_feat_of_clChoice = pd.concat((datafr_scaled[signif_feat], datafr_scaled["class"]), axis=1)

    return(datafr_scaled_with_import_feat_of_clChoice)
    


def import_feat_of_multiple_classifiers(datafr_scaled):
    """
    This function takes a scaled data frame and returns a scaled data frame with 
    only the 20 most important features as columns (plus the "class" column) 
    according to all classifiers included in the "clChoice" variable
    and "classifierChoice" function.
    """
    
    clChoice = ["RF", "XG", "EX_T", "ADA", "GB"]
    ## In case of the addition of another classifier, we have to add the respective code on
    ## "classifierChoice" fuction and the respective abbrevation on this function.
    
    sign_feat_200_5cl = []
    
    for choice in clChoice:
        print(f"the current classifier is {choice}")
        signif_feat_list = HoldOut_sign_feat(datafr_scaled.iloc[:, :-1], datafr_scaled.iloc[:, -1], choice)
        sign_feat_200_5cl.append(signif_feat_list)

    signif_feat_5_cl, signif_counts_5_cl = most_frequent_feat(sign_feat_200_5cl)

    df_import_feat_multiple_classif = pd.concat((datafr_scaled[signif_feat_5_cl], datafr_scaled["class"]), axis=1)

    return(df_import_feat_multiple_classif)

