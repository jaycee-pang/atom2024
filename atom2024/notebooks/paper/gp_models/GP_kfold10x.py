import numpy as np 
import pandas as pd
import torch 
import sys
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks')
from GP_functions import * 
from training_functions import * 
from dataset import * 

if __name__ == '__main__': 
    neks = ['NEK2_binding', 'NEK2_inhibition', 'NEK3_binding', 'NEK5_binding','NEK9_binding','NEK9_inhibition']
    feats = ['MOE','MFP']
    samps = ['none_scaled','UNDER', 'SMOTE', 'ADASYN']
    kernel_type = ['RBF','matern' ]
    train_results = []
    test_results = []
    final_cols=['model','NEK','strategy','feat_type','kernel_type', 'cm','recall', 'specificity', 'accuracy', 'precision', 
                    'f1', 'ROC_AUC', 'MCC', 'balanced_accuracy', 'fold','iteration']
    data_path = '/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/featurized/'
    folds=['fold1','fold2','fold3','fold4','fold5']
    rng = np.random.default_rng(seed=42) # Create a Generator object with a seed 
    numbers = rng.integers(low=0, high=1e6, size=10)  # Generate random numbers
    print(numbers)
    # for i,num in enumerate(numbers):
    for nek in ['NEK2_binding']: 
        for feat in ['MOE']: 
            split_df = pd.read_csv(f'{data_path}{nek}_{feat}_none_scaled.csv')
            train=split_df[split_df['subset']=='train'] 
            folded_train_df = create_folds(train,numbers[0]) # 5 fold split (validation models) in this iteration 
            for fold in folds: # then use these 5 folds for train/validation 
                kfold_df=label_subsets(folded_train_df, fold, 'test') 
                if feat == 'MOE': 
                    featurized_df = featurize(feat_type='MOE',data_path=None, filename=None,moe_path=None, moe_file=None, moe_df=folded_train_df,df=kfold_df) 
                else: 
                    featurized_df = featurize(feat_type='MFP', df=kfold_df,mfp_radius=2, nBits=2048)

                for samp in ['SMOTE']:
                    if samp == 'UNDER': 
                        sampled_df = under_sampling(data_path=None,filename=None,df=featurized_df)  
                    elif samp == "SMOTE" or samp == "ADASYN": 
                        sampled_df=over_sampling(data_path=None,filename=None,df=featurized_df, sampling=samp) 
                    elif samp == 'none_scaled': 
                        sampled_df = featurized_df 
                        
                    print(f'{nek} {feat} {samp} {fold} (it: 0)')
                    id_cols = ['NEK', 'compound_id','base_rdkit_smiles','subset', 'active'] 
                    trainX, trainy, testX, testy = make_torch_tens_float(sampled_df)
                    for kernel in kernel_type: 
                        train_perf, test_perf, model, likelihood= save_results(trainX, trainy, testX, testy, f'{nek}', kernel, n_iterations=300, n_samples=100)   
                        for i, df in enumerate(list([train_perf, test_perf])): 
                            df['NEK'] = nek
                            df['feat_type']=feat 
                            df['strategy']=samp
                            df['kernel_type']=f'GP_{kernel}'
                            df['fold']=fold
                            df['iteration']='testing JP/RS'
                            df['model'] =f'{nek}_{feat}_{samp}_{kernel}_{fold}_iteration0'
            
                        train_results.append(train_perf.iloc[[0]][final_cols].values.flatten())
                        test_results.append(test_perf.iloc[[0]][final_cols].values.flatten())

    all_train =  pd.DataFrame(train_results,columns=final_cols)
    all_train['modeling_type'] = 'GP' 
    all_train['set'] = 'foldvalidation' 
    all_train.to_csv(f'GP_train_results_all_NEK_kfold_val_10x.csv', index=False)    

    all_test =  pd.DataFrame(test_results,columns=final_cols)
    all_test['modeling_type'] = 'GP' 
    all_test['set'] = 'foldvalidation' 
    all_test.to_csv(f'GP_test_results_all_NEK_kfold_val_10x.csv', index=False)                 
                        