import pyforest
import sys
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/')
from updated_RF import *
from dataset import * 
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
    data_path = '/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/featurized/'
    # results_dir='/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/results/RF_results/'
    neks = ['NEK2_binding','NEK2_inhibition','NEK3_binding','NEK5_binding','NEK9_binding','NEK9_inhibition']
    samplings =['none_scaled','UNDER','SMOTE','ADASYN'] 
    feats=['MOE','MFP'] 
    RF_types = ['RF','RF_BCW','BRFC','BRFC_BCW']
    train_results = []
    test_results=[]
    final_cols=['model','NEK','strategy','feat_type','RF_type', 'cm','recall', 'specificity', 'accuracy', 'precision', 
                'f1', 'ROC_AUC', 'MCC', 'balanced_accuracy']
    rng = np.random.default_rng(seed=42) # Create a Generator object with a seed 
    numbers = rng.integers(low=0, high=1e6, size=10)  # Generate random numbers
    print(numbers)
    
    for nek in ['NEK2_binding']: 
        for feat in ['MFP']: 
            for i, num in enumerate(numbers):
                
                split_df = pd.read_csv(f'{data_path}{nek}_{feat}_none_scaled.csv')
                train=split_df[split_df['subset']=='train'] 
                folded_train_df = create_folds(train,num) # 5 fold split (validation models) in this iteration 
        
                for fold in ['fold1']: # then use these 5 folds for train/validation 
                    kfold_df=label_subsets(folded_train_df, fold, 'test') 
                    if feat == 'MOE': 
                        featurized_df = featurize(feat_type='MOE',data_path=None, filename=None,moe_path=None, moe_file=None, moe_df=folded_train_df,df=kfold_df) 
            
                    else: 
                        featurized_df = featurize(feat_type='MFP', df=kfold_df,mfp_radius=2, nBits=2048)

                    for samp in ["none_scaled",'UNDER', 'SMOTE']:
                        if samp == 'UNDER': 
                            sampled_df = under_sampling(data_path=None,filename=None,df=featurized_df)  
                        elif samp == "SMOTE" or samp == "ADASYN": 
                            sampled_df=over_sampling(data_path=None,filename=None,df=featurized_df, sampling=samp) 
                        elif samp == 'none_scaled': 
                            sampled_df = featurized_df 
                            
                        root_name = f'{nek}_{feat}_{samp}'
                        print(f'{nek} {feat} {samp} {fold} (it: {i})')
                        id_cols = ['NEK', 'compound_id','base_rdkit_smiles','subset', 'active'] 
                        trainX, train_y, testX, test_y=get_arrays(file_path=None, root_name=None, df=sampled_df,nonfeat_cols=id_cols)
                        model = rf_models(trainX, train_y, testX, test_y, 'RF', {}, True)  # make sure dict and doesn't go to default RF version
                        train_df = gather_rf_results(model, trainX, train_y)
                        test_df = gather_rf_results(model, testX, test_y)
                        print()
                        for this_df in [train_df,test_df]: 
                            this_df['model'] = f'{nek} {feat} {samp} {fold} (it: {i}'
                            this_df['NEK'] =nek
                            this_df['feat_type'] = feat
                            this_df['strategy'] = samp
                            this_df['RF_type'] = 'RF'
                        
                     

                        train_df['fold'] = fold 
                        test_df['fold']=fold
                        train_df['iteration']=i
                        test_df['iteration']=i
                        train_results.append(train_df.iloc[[0]][final_cols].values.flatten())
                        test_results.append(test_df.iloc[[0]][final_cols].values.flatten())

                        


all_test =  pd.DataFrame(test_results,columns=final_cols)

all_test['modeling_type'] = 'RF' 
all_test['set'] = 'foldvalidation' 
all_test.to_csv(f'RF_test_results_all_NEK_kfold_val_10x.csv', index=False)                 
                    
                        
                    