import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle 
import sklearn

import imblearn as imb

from sklearn.metrics import confusion_matrix
import itertools

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys
from RF_atomver import *


def get_arrays(file_path, df_filename, filename_type=None, save=False,yesPrint=False):
    """use dataframes to get trainX, trainy, testX, testy out. Optional: save those files to csv
    file_path: directory
    df_filename: dataframe NEK#_binding_moe_{sampling}_df.csv (sampling: scaled, UNDER, SMOTE, ADASYN)
    split dataframe to train and test, and x and y
    save: bool, option to save splits to separate csv files (train X, train y, test X, test y) 
    returns: numpy arrays train X, train y, testX, test y"""
    df = pd.read_csv(file_path+df_filename)
    train_df= df[df['subset']=='train']
    test_df = df[df['subset']=='test']
    train_y = train_df['active'].to_numpy().reshape(-1)
    test_y=test_df['active'].to_numpy().reshape(-1)
    train_x_df = train_df.drop(columns='active')
    test_x_df = test_df.drop(columns='active')
    train_x_df = train_df.drop(columns='active')
    test_x_df = test_df.drop(columns='active')
    trainX = train_x_df.select_dtypes(include='number').to_numpy()
    testX = test_x_df.select_dtypes(include='number').to_numpy()

    if yesPrint: 
        print(f'train X shape: {trainX.shape}, y: {train_y.shape}, test X: {testX.shape}, y:{test_y.shape}')
    if (save and filename_type is not None): 
        trainxdf = pd.DataFrame(trainX)
        trainxdf.to_csv(file_path+filename_type+'_trainX.csv', index=False)
        trainy_df = pd.DataFrame(train_y)
        trainy_df.to_csv(file_path+filename_type+'_train_y.csv', index=False) 
        testxdf = pd.DataFrame(testX)
        testxdf.to_csv(file_path+filename_type+'_testX.csv', index=False)
        testy_df = pd.DataFrame(test_y)
        testy_df.to_csv(file_path+filename_type+'_test_y.csv', index=False) 
        
    return trainX, train_y, testX, test_y

if __name__ == '__main__':
    grid_2 = {
        'n_estimators': np.linspace(100, 2000, 5, dtype=int),
        'criterion': ['gini','entropy'],
        'max_features': [7, 17, 27, 36, 46, 56, 100, 300, 685, 2048]

    }
    neks = ['NEK2_binding', 'NEK2_inhibition', 'NEK3_binding', 'NEK5_binding', 'NEK9_binding', 'NEK9_inhibition'] 
    nek_list = ["2", "3", "5", "9"]
    nektype = ['binding','inhibition']
    feat_types = ['moe', 'mfp']
    samplings = ['scaled', 'UNDER' , 'SMOTE', 'ADASYN']
    model_types = ['RF','RF_BCW', 'BRFC', 'BRFC_BCW']
    count = 0 
    for n in nek_list:
        for i in nektype: 
            for feat in feat_types: 
                for samp in samplings:  
                    for rf in model_types:  # RF, RF_BCW, BRFC, BRFC_BCW
                        if i == 'binding':
                            data_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK{n}/bind/'
                        
                            df_name = f'NEK{n}_{i}_{feat}_{samp}_df.csv'
                            count+=1
                            trainX, trainy, testX, testy = get_arrays(data_path, df_name)
                            grid_search = find_best_models(trainX, trainy, testX, testy, rf, {}, grid_2, verbose_val=2)
        
                            model_name = f'NEK{n}_{i}_{feat}_{samp}_{rf}_GS'
                            print(f'{count}. {model_name}')
                            
                            with open(f'{model_name}.pkl', 'rb') as f:
                                pickle.dump(grid_search, f) 
                            
                        if n in ['2', '9'] and i == 'inhibition': 
                            data_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK{n}/inhib/'
                            df_name = f'NEK{n}_{i}_{feat}_{samp}_df.csv'
                            trainX, trainy, testX, testy = get_arrays(data_path, df_name)
        
                            grid_search = find_best_models(trainX, trainy, testX, testy, rf, {}, grid_2, verbose_val=2)
                            count+=1
                            model_name = f'NEK{n}_{i}_{feat}_{samp}_{rf}_GS'
                            print(f'{count}. {model_name}')
                            
                            with open(f'{model_name}.pkl', 'rb') as f:
                                pickle.dump(grid_search, f)   
                    
                    
    


    