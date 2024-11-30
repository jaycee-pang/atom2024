import pyforest
from updated_RF import *
from imblearn.over_sampling import SMOTEN, ADASYN, SMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score,precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix,matthews_corrcoef

from rdkit import Chem
from rdkit.Chem import AllChem


import math
import torch
import numpy as np
# import gpytorch
import pandas as pd
import seaborn as sns
import os
import pickle
import shutil
import matplotlib 
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import KFold

import imblearn as imb
# print("imblearn version: ",imblearn.__version__)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import itertools

# from scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys


from sklearn.model_selection import GridSearchCV

if __name__ == '__main__': 
    kf_path = '/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/'
    val_models='/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/val_models/'
    results_dir='/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/results/'
    neks = ['NEK2_binding','NEK2_inhibition','NEK3_binding','NEK5_binding','NEK9_binding','NEK9_inhibition']
    samplings =['none_scaled','UNDER','SMOTE','ADASYN'] 
    feats=['MOE','MFP'] 
    RF_types = ['RF','RF_BCW','BRFC','BRFC_BCW']
    folds=['fold1','fold2','fold3','fold4','fold5'] 
    kf_path = '/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/'
val_models='/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/val_models/'
results_dir='/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/k_fold/validation/results/'
neks = ['NEK2_binding','NEK2_inhibition','NEK3_binding','NEK5_binding','NEK9_binding','NEK9_inhibition']
samplings =['none_scaled','UNDER','SMOTE','ADASYN'] 
feats=['MOE','MFP'] 
RF_types = ['RF','RF_BCW','BRFC','BRFC_BCW']
folds=['fold1','fold2','fold3','fold4','fold5'] 
train_results = []
test_results=[]
final_cols=['model','NEK','strategy','feat_type','RF_type','fold', 'cm','recall', 'specificity', 'accuracy', 'precision', 
            'f1', 'ROC_AUC', 'MCC', 'balanced_accuracy']
for nek in neks: 
    for feat in feats: 
        for samp in samplings:
            for fold in folds: 
                root_name = f'{nek}_{feat}_{samp}_{fold}_validation'
                
                trainX=pd.read_csv(f'{kf_path}{root_name}_trainX.csv').iloc[:, :-1]
                train_y=pd.read_csv(f'{kf_path}{root_name}_train_y.csv').to_numpy().reshape(-1)
                testX=pd.read_csv(f'{kf_path}{root_name}_testX.csv').iloc[:, :-1]
                test_y=pd.read_csv(f'{kf_path}{root_name}_test_y.csv').to_numpy().reshape(-1)
                for rf in RF_types: 
                    model_name = f'{nek}_{feat}_{samp}_{rf}_{fold}'
                    print(model_name) 
                    model = rf_models(trainX, train_y, testX, test_y, rf, {}, True)  # make sure dict and doesn't go to default RF version

                    # with open(f'{model_pickle_dir}{model_name}.pkl', 'wb') as f: 
                    #     pickle.dump(model, f) 
                    # add cm and other metrics to this function 
                    train_df = gather_rf_results(model, trainX, train_y)
                    test_df = gather_rf_results(model, testX, test_y)
                    print()
            
                    
                    for this_df in [train_df,test_df]: 
                        this_df['model'] = model_name
                        # this_df= add_cm(this_df)
                        this_df['NEK'] =nek
                        this_df['feat_type'] = feat
                        this_df['strategy'] = samp
                        this_df['RF_type'] = rf
                    
                    train_df.to_csv(f'{results_dir}{model_name}_train.csv',index=False) 
                    test_df.to_csv(f'{results_dir}{model_name}_test.csv',index=False) 
                    train_df['fold'] = fold 
                    test_df['fold']=fold
                    train_results.append(train_df.iloc[[0]][final_cols].values.flatten())
                    test_results.append(test_df.iloc[[0]][final_cols].values.flatten())

    all_train =  pd.DataFrame(train_results,columns=final_cols)
    all_test =  pd.DataFrame(test_results,columns=final_cols)
    all_train['modeling_type'] = 'RF' 
    all_train['set'] = 'foldvalidation' 
    all_test['modeling_type'] = 'RF' 
    all_test['set'] = 'foldvalidation' 
    all_train.to_csv('RF_train_results_all_NEK_kfold_val.csv', index=False) 
    all_test.to_csv('RF_test_results_all_NEK_kfold_val.csv', index=False)                 
                      
                        
                    

