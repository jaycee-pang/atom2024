from RF_GSCV import *
from sklearn.model_selection import KFold
def make_splits(full_df, split_path, name):
    """ Make balanced ratio splits for majority/minority (assuming minority class is the active class)
        and save to a Df with column 'fold'
    full_df is the 1uM 50 pct MOE descriptor dataframe 
    split_path: where to save the final concated df 
    name: add NEK# and indicate binding/inhibition
    """
    print(f'dataset shape:  {full_df.shape}')
    print(full_df.active.value_counts())
    # assumes active class 1 is the minority 
    num_gap = (full_df.loc[full_df['active']==0].shape[0])-(full_df.loc[full_df['active']==1].shape[0])
    num_minority = full_df.loc[full_df['active']==1].shape[0]
    print(f'Number of minority samples: {num_minority}')
    # separate majority and minority 
    df_major = full_df[full_df['active']==0]
    df_minor = full_df[full_df['active']==1]
    print(f'Majority shape: {df_major.shape}')
    print(f'Minority shape: {df_minor.shape}')
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # majority
    for i, (_, v_ind) in enumerate(kf.split(df_major)):
        df_major.loc[df_major.index[v_ind], 'fold'] = f"fold{i+1}"
    
    # minority
    for i, (_, v_ind) in enumerate(kf.split(df_minor)):
        df_minor.loc[df_minor.index[v_ind], 'fold'] = f"fold{i+1}"
    print(f'Majority fold counts: {df_major["fold"].value_counts()}')
    print(f'Minority fold counts: {df_minor["fold"].value_counts()}')   
    # concat and save to file 
    all_fold_df = pd.concat([df_major, df_minor])
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    print('after major+minor concat:')
    
    
    print(f'all_fold shape: {all_fold_df.shape}')
    print(f'all_fold active value counts: {all_fold_df.active.value_counts()}')
    split_path = os.path.join(split_path, '')

    all_fold_df.to_csv(split_path+name+"_JP_splits_1_uM_min_50_pct_5fold_random_imbalanced.csv", index=False)
    return all_fold_df

def get_datasplits(save_path, name, all_fold_df=None, all_fold_path=None): 
    """ Split the into different train/test splits based on fold # and save 
        final trainx, trainy, testx, testy to csv 
    save_path: where to save the train/test splits 
    name: add NEK# and binding/inhibition
    all_fold_df: pass in the kfold split ff 
    all_fold_path: path to the kfold split df 
    @returns: train_x_df, train_y_df, test_x_df, test_y_df 
        should contain balanced ratios across the different folds 
    """
    foldAll = ["fold1","fold2","fold3","fold4","fold5"]
    if (all_fold_df is not None and not all_fold_df.empty): 
        random_df = all_fold_df
    elif(split_path is not None): 
        random_df = pd.read_csv(all_fold_path)
    else: 
        raise ValueError('Provide a valid dataframe or the path to the dataframe')
    moe_columns = random_df.columns[3:] # all_fold_MOE [3:] # compound_id, base_rdkit_smiles, active
    moe_columns = moe_columns[:-1]
    # print(moe_columns)
    # use fold 0 as test, folds 1-4 as the train set 
    for fold in foldAll: 
        test_moe_df = random_df.loc[random_df['fold'] == fold]
        train_moe_df = random_df.loc[random_df['fold'] != fold]
        test_x_df = test_moe_df[moe_columns]
        test_y_df = test_moe_df['active']
        train_x_df = train_moe_df[moe_columns]
        train_y_df = train_moe_df['active']
        print(f'train moe df shape: {train_moe_df.shape}, test moe df shape: {test_moe_df.shape}')
        print(f'final train x df shape: {train_x_df.shape}, final train y df shape: {train_y_df.shape}')
        print(f'final test x df shape: {test_x_df.shape}, final test y df shape: {test_y_df.shape}')
        print(f'final train y value counts: {train_y_df.value_counts()}, final test y value counts: {test_y_df.value_counts()}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, '')
        train_x_df.to_csv(save_path+name+"_random_"+fold+"_trainX.csv", index=False)
        train_y_df.to_csv(save_path+name+"_random_"+fold+"_trainY.csv", index=False)
        test_x_df.to_csv(save_path+name+"_random_"+fold+"_testX.csv", index=False)
        test_y_df.to_csv(save_path+name+"_random_"+fold+"_testY.csv", index=False)
    return train_x_df, train_y_df, test_x_df, test_y_df
    

    