import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import os

import shutil
import sklearn
from sklearn.model_selection import KFold
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel

from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, f1_score, roc_curve,precision_recall_curve, auc
# from scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

import sys
sys.path.append('/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks')
from RF_GSCV import * # RF_GSCV contains the calculate metrics function to get the TP, TN, FP, FN scores 
from RF_atomver import prediction_type 


class DirichletGPModel(ExactGP):
    """
    A Dirichlet Gaussian Process (GP) model for multi-class classification.
    This model uses a Gaussian Process with a Dirichlet prior to handle multi-class classification tasks.
    It extends the ExactGP class from GPyTorch, a library for Gaussian Processes in PyTorch.
    Attributes:
        mean_module (gpytorch.means.ConstantMean): The mean module for the GP, initialized with a constant mean function for each class.
        covar_module (gpytorch.kernels.ScaleKernel): The covariance module for the GP, using a scaled RBF kernel for each class.

    Args:
        train_x (torch.Tensor): Training data features.
        train_y (torch.Tensor): Training data labels.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
        num_classes (int): The number of classes for the classification task.
    """
    def __init__(self, train_x, train_y, likelihood, num_classes,kernal):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        if kernal == 'matern': 
            self.covar_module = ScaleKernel(MaternKernel(nu=0.5, batch_shape=torch.Size((num_classes,))),
                batch_shape=torch.Size((num_classes,))
            )
        elif kernal == 'RBF': 
            self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),)

        else: 
            print('invalid')
        
    def forward(self, x):
        """
        Forward pass through the GP model.
        Args:
            x (torch.Tensor): Input data features.
        Returns:
            gpytorch.distributions.MultivariateNormal: The multivariate normal distribution representing the GP posterior.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class Trainer: 
    def __init__(self,model, likelihood, iterations): 
        self.model = model
        self.likelihood = likelihood 
        smoke_test = ('CI' in os.environ)
        self.n_iterations = 2 if smoke_test else iterations
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        self.loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
    def train(self, train_x, train_y): 
        self.model.train()
        self.likelihood.train()
        predictions = [] 
        for i in range(self.n_iterations): 
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.loss_fn(output, self.likelihood.transformed_targets).sum()
            loss.backward()
            if (i%10==0): 
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.n_iterations, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.mean().item(),
                    self.model.likelihood.second_noise_covar.noise.mean().item()
                ))
             
            self.optimizer.step() 
    def predict(self, input): 
        """
        Make predictions using the GP model.

        Args:
            input (torch.Tensor): The input data for making predictions.
        
        Returns:
            dist (gpytorch.distributions.MultivariateNormal): The distribution representing the GP posterior.
            observed_pred (gpytorch.distributions.MultivariateNormal): The predicted distribution considering the likelihood.
            pred_means (torch.Tensor): The means of the predicted distributions.
            class_pred (torch.Tensor): The predicted class labels.
        """
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            dist = self.model(input)     # output distribution
            pred_means = dist.loc          # means of distributino 
            observed_pred = self.likelihood(self.model(input))    # likelihood predictions mean and var  

            class_pred = self.model(input).loc.max(0)[1]
            
        return dist, observed_pred, pred_means, class_pred
    

    def evaluate(self, x_input, y_true): 
        """
        Evaluate the GP model.

        Args:
            x_input (torch.Tensor): The input data features.
            y_true (torch.Tensor): The true labels for the input data.
        
        Returns:
            y_pred (numpy.ndarray): The predicted class labels.
        """
        y_pred = self.model(x_input).loc.max(0)[1].numpy()
        
        return y_pred
    
    def calculate_metrics(y_true, y_pred): 
        
        # return tp, tn, fp, fn
        y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
        y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return tp, tn, fp, fn

    def gp_results(self, x_input, y_true, plot_title=None): 
        """
        Calculate evaluation metrics and print results.

        Args:
            x_input (torch.Tensor): The input data features.
            y_true (torch.Tensor or numpy.ndarray): The true labels for the input data.
            plot_title (str, optional): The title for the confusion matrix plot.
        
        Returns:
            dict: A dictionary containing evaluation metrics and confusion matrix components.
        """
        y_pred = self.evaluate(x_input, y_true) 
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy().reshape(-1)
        # plot_confusion_matrix(y_true, y_pred, ['0','1'], title=plot_title)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        dist = self.model(x_input)     # get predicted distributions 
        pred_means = dist.loc          # means for predicted dist  

        recall = recall_score(y_true, y_pred)
        tp, tn, fp, fn = calculate_metrics(y_true, y_pred) 
       
        specificity = tn / (tn + fp) 
        cm = confusion_matrix(y_true, y_pred)
        cm_flattened = cm.flatten().tolist()
        f1 = f1_score(y_true,y_pred)
        roc_auc = roc_auc_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        bal_acc = balanced_accuracy_score(y_true,y_pred)
        print(f'accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}, cm: {cm}')
        return {'accuracy': accuracy, 'precision': precision,  'recall':recall, 'specificity':specificity, 
                'f1':f1,'ROC_AUC': roc_auc,'MCC': mcc,'balanced_accuracy': bal_acc,'cm': str(cm_flattened),
                'TN': tn, 'FN': fn, 'FP': fp, 'TP': tp }

def make_torch_tens_float(filepath, filename): 
    trainX_df = pd.read_csv(filepath+filename+'_trainX.csv')
    trainy_df = pd.read_csv(filepath+filename+'_train_y.csv')
    testX_df = pd.read_csv(filepath+filename+'_testX.csv')
    testy_df = pd.read_csv(filepath+filename+'_test_y.csv')

    train_x_temp = trainX_df.to_numpy().astype("double") # double 
    test_x_temp = testX_df.to_numpy().astype("double") #double 
    
    train_y_temp = trainy_df.to_numpy().flatten().astype("double") #double 
    test_y_temp = testy_df.to_numpy().flatten().astype("double") #double 
   
    trainX = torch.as_tensor(train_x_temp, dtype=torch.float32)
    trainy = torch.as_tensor(train_y_temp, dtype=torch.float32)
    testX = torch.as_tensor(test_x_temp, dtype=torch.float32)
    testy = torch.as_tensor(test_y_temp, dtype=torch.float32)
    return trainX, trainy, testX, testy



def save_results(trainX, trainy, testX, testy, root_name, kernal, n_iterations=300, n_samples=100):
    """
    Train a Dirichlet Gaussian Process model and save the training and test performance results.

    This function trains a Dirichlet GP model on the given training data, evaluates it on both the training
    and test data, and saves various performance metrics and predictions to pandas DataFrames.

    Args:
        trainX (torch.Tensor): The training data features.
        trainy (torch.Tensor): The training data labels.
        testX (torch.Tensor): The test data features.
        testy (torch.Tensor): The test data labels.
        root_name (str): The root name used for labeling the model in the results.
        n_iterations (int, optional): The number of training iterations. Default is 300.
        n_samples (int, optional): The number of samples for prediction. Default is 100.

    Returns:
        train_perf_df (pd.DataFrame): DataFrame containing performance metrics and predictions for the training data.
        test_perf_df (pd.DataFrame): DataFrame containing performance metrics and predictions for the test data.
    """
    likelihood = DirichletClassificationLikelihood(trainy.long(), learn_additional_noise=True)
    model = DirichletGPModel(trainX, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, kernal=kernal)
    # n_iterations = 300
    trainer = Trainer(model, likelihood, n_iterations)
    trainer.train(trainX, trainy) 
  
    train_dist, train_observed_pred, train_pred_means, train_pred  = trainer.predict(trainX)
    train_results = trainer.gp_results(trainX, trainy)
    test_dist, test_observed_pred, test_pred_means, test_pred  = trainer.predict(testX)
    test_results = trainer.gp_results(testX, testy)
    
    train_observed_pred.mean.numpy()
    train_pred_variance2D = train_observed_pred.variance.numpy()
    test_observed_pred.mean.numpy()
    test_pred_variance2D=test_observed_pred.variance.numpy()
    
    train_pred_samples = train_dist.sample(torch.Size((256,))).exp()
    train_probabilities = (train_pred_samples / train_pred_samples.sum(-2, keepdim=True)).mean(0)

    train_prob_stds = (train_pred_samples / train_pred_samples.sum(-2, keepdim=True)).std(0)

    test_pred_samples = test_dist.sample(torch.Size((100,))).exp()

    test_probabilities = (test_pred_samples / test_pred_samples.sum(-2, keepdim=True)).mean(0)
    test_prob_stds = (test_pred_samples / test_pred_samples.sum(-2, keepdim=True)).std(0)

 
    train_perf_df = pd.DataFrame()
    test_perf_df = pd.DataFrame()
    train_perf_df['mean_pred_class0'] = train_observed_pred.mean.numpy()[0,]
    train_perf_df['mean_pred_class1'] = train_observed_pred.mean.numpy()[1,]
    train_perf_df['y'] = trainy
    train_perf_df['y_pred'] = train_pred_means.max(0)[1]
    train_perf_df['var_pred_class0']=train_observed_pred.variance.numpy()[0,]
    train_perf_df['var_pred_class1']=train_observed_pred.variance.numpy()[1,]
    train_perf_df['pred_prob_class0'] = train_probabilities.numpy()[0,]
    train_perf_df['pred_prob_class1'] = train_probabilities.numpy()[1,]
    train_perf_df['pred_prob_std_class0'] = train_prob_stds.numpy()[0,]
    train_perf_df['pred_prob_std_class1'] = train_prob_stds.numpy()[1,]
    train_perf_df['subset'] = 'train' 
   


    for k, val in train_results.items(): 
        train_perf_df[k] = val
    
    
    test_perf_df['mean_pred_class0'] = test_observed_pred.mean.numpy()[0,]
    test_perf_df['mean_pred_class1'] = test_observed_pred.mean.numpy()[1,]
    test_perf_df['y'] = testy
    test_perf_df['y_pred'] = test_pred_means.max(0)[1]
    test_perf_df['var_pred_class0']=test_observed_pred.variance.numpy()[0,]
    test_perf_df['var_pred_class1']=test_observed_pred.variance.numpy()[1,]
    test_perf_df['pred_prob_class0'] = test_probabilities.numpy()[0,]
    test_perf_df['pred_prob_class1'] = test_probabilities.numpy()[1,]
    test_perf_df['pred_prob_std_class0'] =test_prob_stds.numpy()[0,]
    test_perf_df['pred_prob_std_class1'] = test_prob_stds.numpy()[1,]
    test_perf_df['subset'] = 'test' 

   
    for k, val in test_results.items():
        test_perf_df[k] = val

   
    return train_perf_df, test_perf_df, model, likelihood


if __name__=='__main__': 
    gp_results = "/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/results/GP_results/"
    datapath = '/Users/jayceepang/msse/capstone/atom2024/atom2024/notebooks/paper/datasets/80train_20test/featurized/'
    neks = ['NEK2_binding', 'NEK2_inhibition', 'NEK3_binding', 'NEK5_binding','NEK9_binding','NEK9_inhibition']
    feats = ['MOE','MFP']
    samps = ['none_scaled','UNDER', 'SMOTE', 'ADASYN']
    kernal_type = ['RBF','matern' ]
    final_cols = []
    train_results = []
    test_results = []
    final_cols=['model','NEK','strategy','feat_type','kernel_type', 'cm','recall', 'specificity', 'accuracy', 'precision', 
                    'f1', 'ROC_AUC', 'MCC', 'balanced_accuracy']
    for nek in neks:
        for feat in feats:
            for samp in samps:
                for kernal in kernal_type:
                    root_name = f'{nek}_{feat}_{samp}'
                    trainX, trainy, testX, testy = make_torch_tens_float(datapath,f'{root_name}')
                    train_perf, test_perf, model, likelihood= save_results(trainX, trainy, testX, testy, root_name, kernal, n_iterations=300, n_samples=100)   
                    with open(f'{gp_results}{root_name}_{kernal}.pkl', 'wb') as f: 
                        pickle.dump(model,f)
                    with open(f'{gp_results}{root_name}_{kernal}_likelihood.pkl', 'wb') as f: 
                        pickle.dump(likelihood,f)
                
                    for i, df in enumerate(list([train_perf, test_perf])): 
                        df['NEK'] = nek
                        df['feat_type']=feat 
                        df['strategy']=samp
                    
                        df['kernel_type']=f'GP_{kernal}'
                        df['model'] =f'{root_name}_{kernal}'
                        if i == 0:
                            df.to_csv(f'{gp_results}{root_name}_{kernal}_prod_train.csv', index=False)
                            train_results.append(df.iloc[[0]][final_cols].values.flatten())
                        if i == 1: 
                            df.to_csv(f'{gp_results}{root_name}_{kernal}_prod_test.csv', index=False)
                            test_results.append(df.iloc[[0]][final_cols].values.flatten())
    all_train =  pd.DataFrame(train_results,columns=final_cols)
    all_test =  pd.DataFrame(test_results,columns=final_cols)
    all_train['modeling_type'] = 'GP' 
    all_train['set'] = 'prod' 
    all_test['modeling_type'] = 'GP' 
    all_test['set'] = 'prod' 
    all_train.to_csv(f'{gp_results}GP_prod_train_results_all_NEK.csv', index=False) 
    all_test.to_csv(f'{gp_results}GP_test_results_all_NEK.csv', index=False)                 
                        
                                          
