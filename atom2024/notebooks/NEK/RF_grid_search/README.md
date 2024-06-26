# Running Grid Search for RF 
`RF_atomver.py` contains code for running the grid search 

`RF_gridsearch_allNEK.py`: does grid search for all datasets in a for loop (this can be broken down into separate py files, just ask us if that is better)


- Need to change file paths in this file to where NEK/ folder is (zip file in google https://drive.google.com/file/d/1uX_BhPlho6zpj8Atd3_crHUigkMM6tGC/view?usp=drive_link)
paths to change: 
- line 79:  data_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK{n}/bind/'
- line 93: data_path = f'/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/NEK/NEK{n}/inhib/'


Dataset Directory: 
https://drive.google.com/drive/folders/1kiH2W_UdC3qAb2KuQxbf-uf_yIwPapXu 
- This directory is divided by NEK: NEK2, NEK3, NEK5, NEK9 
NEK2 and NEK9 have subdirectories `bind` and `inhib`. 

All datasets have the file name structure: NEK#_(binding or inhibition)_(feature_type)_(sampling strategy). 

There are two types of dataset files. There are `_df.csv` files contain the fully featurized dataset, identification columns, as well as the subset category 'train' or 'test' to indicate which split that datapoint is in. 
The files ending in 'trainX', 'trainy', 'testX', and 'testy' are the training splits and include only the numerical features (ready for model training or testing). 



`testing_param_grid.csv` is a csv file of the different params we found in our grid searches so far and how we obtained the grid used for the search. 


