import math
import torch
import numpy as np
import gpytorch
import pandas as pd
import seaborn as sns
import os
import pickle
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import KFold

import imblearn as imb
# print("imblearn version: ",imblearn.__version__)
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import itertools

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from rdkit import Chem
from rdkit.Chem import Draw
import sys
# gpytorch 
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024')
# sys.path.append('../')
import utils
from sklearn.model_selection import GridSearchCV
# from VisUtils import *
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks')
from split_data import *
from RF_GSCV import *


if __name__ == "__main__": 

    split_path = '../../../../data/NEK_data_4Berkeley/NEK2/'
    train_x_df = pd.read_csv(split_path+"/NEK2_inhibition_random_fold1_trainX.csv")
    train_y_df= pd.read_csv(split_path+"/NEK2_inhibition_random_fold1_trainY.csv")
    test_x_df= pd.read_csv(split_path+"/NEK2_inhibition_random_fold1_testX.csv")
    test_y_df= pd.read_csv(split_path+"/NEK2_inhibition_random_fold1_testY.csv")
    train_x = torch.from_numpy(train_x_df.to_numpy())
    train_y = torch.from_numpy(train_y_df.to_numpy().reshape(-1))
    test_x = torch.from_numpy(test_x_df.to_numpy())
    test_y = torch.from_numpy(test_y_df.to_numpy().reshape(-1))
    # Scale data
    x_df = pd.concat([train_x_df, test_x_df])
    
    scaling=StandardScaler()
     
    # Use fit and transform method 
    scaling.fit(x_df)
    Scaled_data=scaling.transform(x_df)
    train_x = scaling.transform(train_x_df)
    test_x = scaling.transform(test_x_df) 
    
    train_y = train_y_df.to_numpy().flatten()
    test_y = test_y_df.to_numpy().flatten()
    param_grid = {
    'n_estimators': np.linspace(100, 2000, 3, dtype = int),
    'max_depth': [20, 100, 200, 220],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [2, 5],
    'criterion': ['gini','entropy']
    }
    save_file = 'atom_nek2inhib_rf_basic_best.pkl'
    rf_basicbest = find_best_models(train_x, train_y, test_x, test_y, 'basic RF', {}, param_grid,  save_file, 2)
    save_model(rf_basicbest['best_model'], save_file)
    
    save_file2 = 'atom_nek2inhibition_rf_basicBCW_best.pkl'
    rf_basicBCWbest = find_best_models(train_x, train_y, test_x, test_y, 'balanced class_weight', {}, param_grid,  save_file2, 2)
    save_model(rf_basicBCWbest['best_model'], save_file2)

    save_file3 = 'atom_nek2inhib_BRFC_best.pkl'
    brfc_best = find_best_models(train_x, train_y, test_x, test_y, 'BalancedRandomForestClassifier', {}, param_grid,  save_file3, 2)
    save_model(brfc_best['best_model'], save_file3)

    save_file4 = 'atom_nek2inhib_BRFC_BCW_best.pkl'
    brfc_BCW_best = find_best_models(train_x, train_y, test_x, test_y, 'BalancedRandomForestClassifier', {'class_weight':'balanced', 'bootstrape':'True'}, param_grid,  save_file4, 2)
    save_model(brfc_BCW_best['best_model'], save_file4)
  
    
