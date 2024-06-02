Creating features 
Use standard scalar to scale the given data from 5fold random imbalanced dataframes (MOE features), use sampling techniques on the given MOE datasets, create Morgan fingerprint features and use sampling techniques, and also RDKit features. 
- RandomUndersampler
- SMOTE
- ADASYN


Change the paths, and run the notebooks in order: 
1. `create_moe_feats.ipynb`
2. `create_mfp_feats.ipynb`
3. `create_rdkit_feats.ipynb`

Running each notebook will give a dataframe for each sampling technique for each type of NEK dataset. 

Ex. Running the mfp notebook will generate dataframes for the undersampled, SMOTE, and ADASYN for NEK2 bind, NEK2 inhib, NEK3 bind, NEK5 bind, etc. 
The datasplits: train X, train y, test X, and test y will be generate for each dataset as well. 
The files will be moved to the appropriate directories. The structure is: 
NEK2 

* bind 
* inhib  


NEK3  
* bind

NEK5 
* bind  

NEK9  
* bind
* inhib