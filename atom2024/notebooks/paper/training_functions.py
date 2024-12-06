import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import os

import shutil
import sklearn
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, f1_score, roc_curve,precision_recall_curve, auc
import sys
sys.path.append('/Users/radhi/Desktop/GitHub/atom2024/atom2024/notebooks/')
def calculate_metrics(y_true, y_pred): 
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

# def calculate_metrics(y_true, y_pred): 
        
#     # return tp, tn, fp, fn
#     y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
#     y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     tn = np.sum((y_true == 0) & (y_pred == 0))
#     fp = np.sum((y_true == 0) & (y_pred == 1))
#     fn = np.sum((y_true == 1) & (y_pred == 0))
    
#     return tp, tn, fp, fn

def make_torch_tens_float(filepath=None, filename=None, rootname=None,df=None): 
    if filepath is not None: 
        trainX_df = pd.read_csv(filepath+filename+'_trainX.csv')
        trainy_df = pd.read_csv(filepath+filename+'_train_y.csv')
        testX_df = pd.read_csv(filepath+filename+'_testX.csv')
        testy_df = pd.read_csv(filepath+filename+'_test_y.csv')
    if rootname is not None: 
        df=pd.read_csv(f'{filepath}{rootname}.csv') # NEK2_binding_MOE_none_scaled.csv 
        train=df[df['subset']=='train']
        test=df[df['subset']=='test']
        drop_cols = ['NEK', 'subset','active', 'base_rdkit_smiles','compound_id']
        if 'fold' in df.columns: 
            drop_cols.append('fold')
        trainX_df = train.drop(columns=drop_cols)
        trainy_df=train['active']
        testX_df=test.drop(columns=drop_cols)
        testy_df=test['active']

    train_x_temp = trainX_df.to_numpy().astype("double") # double 
    test_x_temp = testX_df.to_numpy().astype("double") #double 
    
    train_y_temp = trainy_df.to_numpy().flatten().astype("double") #double 
    test_y_temp = testy_df.to_numpy().flatten().astype("double") #double 
   
    trainX = torch.as_tensor(train_x_temp, dtype=torch.float32)
    trainy = torch.as_tensor(train_y_temp, dtype=torch.float32)
    testX = torch.as_tensor(test_x_temp, dtype=torch.float32)
    testy = torch.as_tensor(test_y_temp, dtype=torch.float32)
    return trainX, trainy, testX, testy

