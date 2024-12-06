import pyforest
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTEN, ADASYN, SMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score,precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix,matthews_corrcoef

from rdkit import Chem
from rdkit.Chem import AllChem


import math
import torch

import shutil
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
from sklearn.model_selection import GridSearchCV
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r  took: %2.4f sec' % (f.__name__,  te-ts))
        return result
    return wrap
        
def specificity_score(tn, fp):
    val = (tn/(tn+fp))
    return val
def rf_results(model, x_input, y_labels, printout=False): 
    """Make predictions adn get probabilities
    @params
    model: fitted model (fitted to train set)
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays)
    @returns
    train/test predictions
    train/test accuracies 
    train/test probabilities"""
    pred = model.predict(x_input)
    tp, tn, fp, fn = calculate_metrics(y_labels, pred)
    acc = accuracy_score(y_labels, pred)
    precision = precision_score(y_labels, pred)
    recall = recall_score(y_labels, pred)
    f1 = f1_score(y_labels, pred)
    roc_auc = roc_auc_score(y_labels, pred)
    specificity = specificity_score(tn, fp)
    prob = model.predict_proba(x_input)
    mcc = matthews_corrcoef(y_labels, pred)
    bal_acc = balanced_accuracy_score(y_labels, pred)
    if printout: 
        print(f'accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}')
    return {'prediction': pred,'accuracy' : acc, 'precision' :precision, 'recall' :recall, 
            'specificity': specificity,'f1':f1,'ROC_AUC': roc_auc,
            'MCC': mcc,'balanced_accuracy': bal_acc,'probability' :prob}

@timing
def rf_models(train_x, train_y, test_x, test_y, rf_type, parameters,printout=False):
    """Fit a RF model, make predictions, get probabilities
    @params: 
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays) 
    rf_type: model type: RandomForestClassifier, RandomForestClassifier with class_weight:'balanced', or BalancedRandomForestClassifier
        default is RFC 
    parameters: dict for model params 
    dataset_type: binding or inhibition
    @returns: dict with model, train/test prections and probabilities
    """
    n_estimators = parameters.get('n_estimators', 100)
    random_state = parameters.get('random_state', 42) 
    criterion = parameters.get('criterion', 'gini')
    max_depth = parameters.get('max_depth', 100)
    min_samples_split = parameters.get('min_samples_split', 2) 
    min_samples_leaf = parameters.get('min_samples_leaf', 1) 
    bootstrap = parameters.get('bootstrap', False) 
    max_features = parameters.get('max_features', None) 
    class_weight = parameters.get('class_weight', None)
    
    if (rf_type == 'RF'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    elif (rf_type == 'RF_BCW'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
    elif (rf_type == 'BRFC'):
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    elif (rf_type == 'BRFC_BCW'): 
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')

    else:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    
    model.fit(train_x, train_y)
   
    train_results = rf_results(model, train_x, train_y)
    test_results = rf_results(model, test_x, test_y)
    if printout: 
        print(f"TRAIN: acc: {train_results['accuracy']:.3f}, precision: {train_results['precision']:.3f}, recall: {train_results['recall']:.3f}, spec: {train_results['specificity']:.3f}")
        print(f"TEST: acc: {test_results['accuracy']:.3f}, precision: {test_results['precision']:.3f}, recall: {test_results['recall']:.3f}, spec: {test_results['specificity']:.3f}")
    # return {'model': model, 'train_pred':train_pred, 'test_pred': test_pred,
    #          'train_prob':train_prob, 'test_prob': test_prob}
    return model 

def gather_rf_results(model, x_input, true_labels):
    """Save rf model results to DF"""
    # results = rf_results(model, x_input, true_labels)
    results = rf_results(model, x_input, true_labels)
    # results_df = pd.DataFrame(results)
    # the only reason i did this dictionary was because the probability part wasn't working 
    results_dict = {
        'prediction': results['prediction'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'specificity': results['specificity'],
        'f1': results['f1'], 'ROC_AUC': results['ROC_AUC'],
        'MCC': results['MCC'],'balanced_accuracy': results['balanced_accuracy'],
        'prob_class0': results['probability'][:, 0],
        'prob_class1': results['probability'][:, 1]
    }
    results_df = pd.DataFrame(results_dict)
    results_df['y'] = true_labels
    results_df = add_cm(results_df)
    tp, tn, fp, fn = calculate_metrics(true_labels, results_df['prediction'])
    results_df['TN'] = tn  
    results_df['FN'] = fn
    results_df['FP'] = fp 
    results_df['TP'] = tp     
   
    return results_df 

def calculate_metrics(y_true, y_pred): 
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn
    # y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    # y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    
    # tp = np.sum((y_true == 1) & (y_pred == 1))
    # tn = np.sum((y_true == 0) & (y_pred == 0))
    # fp = np.sum((y_true == 0) & (y_pred == 1))
    # fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp, tn, fp, fn

def add_cm(df): 
    true_labels = df['y'] 
    predictions = df['prediction']
    cm = confusion_matrix(true_labels, predictions )
    cm_flattened = cm.flatten().tolist()
    df['cm'] = [cm_flattened]* len(df)
    return df

