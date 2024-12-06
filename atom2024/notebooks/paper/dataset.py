import pyforest
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTEN, ADASYN, SMOTE
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score,precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix,matthews_corrcoef

from rdkit import Chem
from rdkit.Chem import AllChem

def create_folds(df, num): 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=num)
    for i, (train_index, test_index) in enumerate(skf.split(df, df['active'])):
        df.loc[df.index[test_index],'fold'] = f"fold{i+1}"
    return df

def label_subsets(df, test_fold, label):
   """function to label 'train' or 'test' in the 'subset' column
   to be used to create train/test OR train/val
   fold_df: dataframe with column 'fold'
   test_fold (str): fold to make the test set (the remaining folds will be train)
   label (str): 'test' or 'valididation' 
   """ 
   df['subset'] = df['fold'].apply(lambda x: 'test' if x == test_fold else 'train')
   return df[['NEK','compound_id', 'active', 'base_rdkit_smiles', 'subset']]



def over_sampling(data_path=None,filename=None, df=None, sampling=None, printOut=False):
    """Oversample the datasetes using the SMOTE or ADASYN
    Keeps the feature names and id cols
    file_name (full/absolute path): use the scaled dataframe we just created above 'NEK#_(binding/inhibition)_(MOE/MFP)_none_scaled_df.csv'
    sampling (str): 'SMOTE' or 'ADASYN'
    returns: oversampled dataframe
    """
    id_cols = ['NEK', 'compound_id', 'base_rdkit_smiles','subset', 'active'] 
    if data_path is not None: 
        df = pd.read_csv(data_path+filename) # this is the already scaled ver
    
    feat_cols = list(set(list(df.columns))-set(id_cols))

    # train and test 
    train = df[df['subset']=='train'] 
    test =df[df['subset']=='test'] 
    nek = df['NEK'].iloc[0]
    # separate just id cols
    just_ids = ['NEK', 'compound_id', 'base_rdkit_smiles','subset']
    train_just_ids = train[just_ids]
    test_just_ids = test[just_ids]

    # just feats and 'active'
    trainX = train[feat_cols]
    testX = test[feat_cols]
    
    trainy = train['active']
    testy = test['active']
    
    if sampling == 'ADASYN':
        oversample = ADASYN(random_state=42)
    else: 
        oversample = SMOTE(random_state=42)

    
    trainX_temp, trainy_temp = oversample.fit_resample(trainX.to_numpy(), trainy.to_numpy().reshape(-1))
    if printOut: 
        print(f'train after {sampling}: {trainX_temp.shape}')
    
    trainX_resamp = pd.DataFrame(trainX_temp, columns=feat_cols)
    trainy_resamp = pd.DataFrame(trainy_temp, columns=['active'])

    num_real = len(train)
    num_synthetic = len(trainX_resamp)-num_real
    synthetic_ids = pd.DataFrame({'NEK': [nek] * num_synthetic,
        'compound_id': [f'synthetic_{sampling}_{i}' for i in range(num_synthetic)],
        'base_rdkit_smiles': [f'synthetic_{sampling}'] * num_synthetic,
        'subset': ['train']*num_synthetic}) # ,'active':[1]*num_synthetic}

    real_ids = train_just_ids.reset_index(drop=True)
    combined_ids = pd.concat([real_ids,synthetic_ids], ignore_index=True)
    
    train_resamp = pd.concat([combined_ids, trainX_resamp, trainy_resamp[['active']]], axis=1)

    print(train_resamp.columns[train_resamp.columns.duplicated()])
    test_df_final = pd.concat([test_just_ids.reset_index(drop=True),
                               testX.reset_index(drop=True), testy.reset_index(drop=True)],axis=1)
    
    final_df = pd.concat([train_resamp, test_df_final]).reset_index(drop=True)
    return final_df[list(df.columns)]

def under_sampling(data_path=None,filename=None, df=None):
    id_cols = ['NEK', 'compound_id','base_rdkit_smiles','subset', 'active'] 
    if data_path is not None: 
        df = pd.read_csv(data_path+filename) # this is the already scaled ver
    feat_cols = list(set(list(df.columns))-set(id_cols))
    
    # train and test 
    train = df[df['subset']=='train'] 
    test =df[df['subset']=='test'] 

    # separate just id cols
    just_ids = ['NEK', 'compound_id', 'base_rdkit_smiles','subset']
    train_just_ids = train[just_ids]
    test_just_ids = test[just_ids]

    # just feats and 'active'
    trainX = train[feat_cols]
    testX = test[feat_cols]
    
    trainy = train['active']
    testy = test['active']
    
    undersample = RandomUnderSampler(random_state=42)
    
    trainX_temp, trainy_temp = undersample.fit_resample(trainX.to_numpy(), trainy.to_numpy().reshape(-1))
    
    trainX_resamp = pd.DataFrame(trainX_temp, columns=feat_cols)
    trainy_resamp = pd.DataFrame(trainy_temp, columns=['active'])
    
    train_ids_resamp = train_just_ids.iloc[trainX_resamp.index].reset_index(drop=True)
    train_resamp= pd.concat([train_ids_resamp, trainX_resamp,trainy_resamp], axis=1)
    # train_resamp['subset'] = 'train'

    test_df_final = pd.concat([test_just_ids.reset_index(drop=True),testX.reset_index(drop=True),testy.reset_index(drop=True)],axis=1)
    # test_df_final['subset'] = 'test'
    final_df = pd.concat([train_resamp,test_df_final]).reset_index(drop=True)
    return final_df[list(df.columns)]

def featurize(feat_type,data_path=None, filename=None,moe_path=None, moe_file=None, moe_df=None, df=None,mfp_radius=2, nBits=2048): 
    if (feat_type == 'MOE') and (moe_path is not None) and (data_path is not None): 
        feat_df = create_moe(data_path, filename, moe_path, moe_file)
    elif (feat_type == 'MOE') and (df is not None) and (moe_df is not None): 
        feat_df = create_moe(df=df, moe_df=moe_df)
    elif (feat_type == 'MFP') and (data_path is not None): 
        feat_df = create_mfp(data_path, filename, mfp_radius, nBits)
    elif (feat_type == 'MFP') and (df is not None): 
        feat_df = create_mfp(df=df)
    
    return feat_df

def create_moe(data_path=None, filename=None, moe_path=None, moe_file=None, df=None, moe_df=None):
    """(intended use for already existing dataset)
    This function will use an existing dataframe with smiles column to
    get the features from an existing file (moe_path+moe_file) with the MOE features generated"""
    drop_cols = ['active', 'compound_id']
    id_cols = ['NEK', 'compound_id','base_rdkit_smiles','subset', 'active']
   
    if data_path is not None: 
        df = remove_duplicates(data_path, filename)
    df=df.drop(columns=drop_cols)
    if moe_path is not None: 
        moe_df=remove_duplicates(moe_path,moe_file)
    final_df=moe_df.merge(df, how='outer', on=['base_rdkit_smiles'], suffixes=('_moe_desc', '_og'))
    NEK_col = final_df['NEK_og'] 
    subset_col = final_df['subset_og']
    
    final_df = final_df.loc[:,~final_df.columns.str.endswith(('_moe_desc', '_og'))]
    final_df['NEK']=NEK_col
    final_df['subset']=subset_col
    
    feat_cols = set(list(final_df.columns))-set(id_cols)
    final_order_cols = list(id_cols)+list(feat_cols)
    final_df =final_df[final_order_cols] 
    if 'fold' in final_df.columns: 
        final_df=final_df.drop(columns=['fold']) 
    return final_df
def smiles_to_fps(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            arr = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)

def create_mfp(file_path=None, filename=None, df=None,mfp_radius=2, nBits=2048):
    if file_path is not None: 
        df = pd.read_csv(file_path+filename)
    
    id_cols = ['NEK', 'compound_id','base_rdkit_smiles','subset', 'active'] 
    
   
    smiles = df['base_rdkit_smiles']
    mfp_feats = smiles_to_fps(smiles,mfp_radius,nBits)
    mfp_df = pd.DataFrame(mfp_feats)
    # if mfp_df['base_rdkit_smiles'].isnull().any():
    #     print("Warning: Missing values found in 'base_rdkit_smiles' column in df.")
    valid_smiles = smiles[smiles.apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    feat_cols = set(list(mfp_df.columns))-set(id_cols)
    final_order_cols = list(id_cols)+list(feat_cols)

    mfp_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    final_df = pd.concat([df,mfp_df],axis=1)

    final_df = final_df[final_order_cols]
    return final_df


def get_arrays(file_path=None, root_name=None, df=None,nonfeat_cols=None): 
    if file_path is not None: 
        df=pd.read_csv(f'{file_path}{root_name}.csv')
    train=df[df['subset']=='train']
    test=df[df['subset']=='test']
    train_y = train['active'].to_numpy().reshape(-1) 
    test_y =test['active'].to_numpy().reshape(-1) 
    trainX = train.drop(columns= nonfeat_cols) 
    testX = test.drop(columns= nonfeat_cols) 
    return trainX, train_y, testX, test_y