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
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/')
# from split_data import *
from RF_atomver import *

def get_arrays(file_path, df_filename, filename_type=None, save=False):
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
    
    print(f'train X shape: {trainX.shape}, y: {train_y.shape}, test X: {testX.shape}, y:{test_y.shape}')
    if (save and filename_type is not None): 
        trainxdf = pd.DataFrame(trainX)
        trainxdf.to_csv(file_path+filename_type+'_trainX.csv', index=False)
        # train_x_df.to_csv(filename_type+'_trainX.csv', index=False)
        trainy_df = pd.DataFrame(train_y)
        trainy_df.to_csv(file_path+filename_type+'_train_y.csv', index=False) 
        # test_x_df.to_csv(filename_type+'_testX.csv', index=False)
        testxdf = pd.DataFrame(testX)
        testxdf.to_csv(file_path+filename_type+'_testX.csv', index=False)
        testy_df = pd.DataFrame(test_y)
        testy_df.to_csv(file_path+filename_type+'_test_y.csv', index=False) 
        
    return trainX, train_y, testX, test_y
if __name__ == '__main__':
    feat_types = ['moe', 'mfp']
    samplings = ['SMOTE']
    model_types = ['RF','BRFC']
    nektype = ['binding','inhibition']
    param_dist = {
        'n_estimators': np.linspace(100, 2000, 5, dtype = int),
        'max_depth': [20, 100, 200, 250],
        'min_samples_split': [2, 3,4,5],
        'min_samples_leaf': [2, 3, 4,5],
        'criterion': ['gini','entropy'],
        'class_weight':[ None, 'balanced','balanced_subsample']

    } 
    for i in nektype: 
        for j in feat_types: 
            for k in samplings: 
                for l in model_types: 
                    print(f'{i}, {j}, {k}, {l}')
                    if (i == 'binding'):
                        file_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK2/bind/'
                    if (i == 'inhibition'):
                        file_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK2/inhib/'
                    df_name = f'NEK2_{i}_{j}_{k}_df.csv'
                    trainX, trainy, testX, testy = get_arrays(file_path,df_name)
                    grid_search = find_best_models(trainX, trainy, testX, testy, l, {}, param_dist,  verbose_val=2)
                    model_name = f'NEK2_{i}_{j}_{k}_{l}_GS.pkl'
                    with open(f'{model_name}', 'wb') as f: 
                        pickle.dump(grid_search, f)  