import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, PrecisionRecallDisplay

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=18)
    fig.tight_layout();
    # return ax

# Plot Heatmap
def plot_heatmap(data, title="Heatmap", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plots a heatmap for the given data.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt="d", cmap='BuPu')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Plot t-SNE
def plot_tsne(features, labels, title='t-SNE'):
    """
    Plots a t-SNE visualization for the given features and labels.
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=sns.color_palette("hsv", 10),
        legend="full",
        alpha=0.3
    )
    plt.title(title)

# Plot Histogram
def plot_hist(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram for the given data.
    """
    plt.hist(data, bins=bins, alpha=0.6, color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Plot Histogram for True Positives and True Negatives
def plot_hist_tp_tn(y_true, y_pred_prob, threshold=0.5):
    """
    Plots histograms for true positives and true negatives based on a given threshold.
    """
    tp = y_pred_prob[(y_true == 1) & (y_pred_prob >= threshold)]
    tn = y_pred_prob[(y_true == 0) & (y_pred_prob < threshold)]

    plt.hist(tp, bins=30, alpha=0.5, label='True Positives', color='green')
    plt.hist(tn, bins=30, alpha=0.5, label='True Negatives', color='red')
    plt.legend(loc='upper right')
    plt.show()

# Plot Confusion Matrix Distribution
def plot_cm_dist(y_true, y_pred, title="Confusion Matrix Distribution"):
    """
    Plots distributions for each quadrant of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot KDE for Confusion Matrix Distribution
def plot_cm_dist_kde(y_true, y_pred_prob, title="KDE for Confusion Matrix Distribution"):
    """
    Plots Kernel Density Estimation (KDE) for true positives and false positives.
    """
    sns.kdeplot(y_pred_prob[y_true == 1], label='True Positives', shade=True)
    sns.kdeplot(y_pred_prob[y_true == 0], label='False Positives', shade=True)
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_class_and_probability_grids(y_true, probabilities, title_prefix=''):
    """
    Plots grids for actual classes and prediction probabilities side by side.

    Parameters:
    - y_true: Actual class labels (numpy array).
    - probabilities: Prediction probabilities (assumed to be a tensor, requires .numpy() method).
    - title_prefix: Optional prefix for the plot titles.
    """
    # Ensure `y_true` is a numpy array and convert `probabilities` to numpy array
    actual_classes = np.array(y_true)
    probabilities_np = probabilities.numpy() if hasattr(probabilities, 'numpy') else probabilities

    # Calculate the side length of the grid for square arrangement
    num_samples = actual_classes.shape[0]
    side_length = int(np.ceil(np.sqrt(num_samples)))

    # Prepare actual class grid data
    actual_grid_data = np.full((side_length, side_length), np.nan)
    actual_grid_data.flat[:num_samples] = actual_classes

    # Assume probabilities_np is structured with probabilities in the second axis
    active_probabilities = probabilities_np[:, 1] if probabilities_np.ndim > 1 else probabilities_np
    prediction_grid_data = np.full((side_length, side_length), np.nan)
    prediction_grid_data.flat[:num_samples] = active_probabilities

    # Plotting both grids side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Plot actual class grid
    im0 = axs[0].imshow(actual_grid_data, cmap='RdYlGn', origin='lower', aspect='equal')
    fig.colorbar(im0, ax=axs[0], ticks=[0, 1], label='Actual Class')
    axs[0].set_title(f'{title_prefix}Actual Class Grid')
    axs[0].axis('off')

    # Plot prediction probabilities grid
    im1 = axs[1].imshow(prediction_grid_data, cmap='RdYlGn', origin='lower', aspect='equal')
    fig.colorbar(im1, ax=axs[1], label='Probability of Active Class')
    axs[1].set_title(f'{title_prefix}Prediction Probabilities Grid')
    axs[1].axis('off')

    plt.show()

def plot_kde(observed_pred, title): 
    plt.figure(figsize=(8, 6))
    var = observed_pred.variance.numpy().tolist()
    class0_var = observed_pred.variance[0].numpy() 
    class1_var = observed_pred.variance[1].numpy() 
    
    sns.kdeplot(class0_var, label=f'Class 0')
    sns.kdeplot(class1_var, label=f'Class 1')
    plt.xlabel('Variance')
    plt.ylabel('Density')
    plt.title(f'{title} KDE Variances for Each Class')
    plt.legend()
    plt.grid(True)
    plt.show();

def look_at_data(filepath):
    """5-fold on majority and minority separately, then concat into one df""" 
    df = pd.read_csv(filepath)

    print("Dataset shape:",df.shape)
    print(df.active.value_counts())
    print(df['fold'].unique())
    num_gap = (df.loc[df['active']==0].shape[0]) - (df.loc[df['active']==1].shape[0])
    print("\nDifference in class sample sizes: ",num_gap)

    num_minority = df.loc[df['active']==1].shape[0]
    print("Number of minority samples: ",num_minority)
    # print(df.describe())
    print(f"active/inactive: {df['active'].value_counts()}")
    print(f"active/inactive: {df['active'].value_counts()}")
    counts_per_fold = df.groupby('fold')['active'].value_counts()
    print(counts_per_fold)
    return df


def plot_cm_dist_kdedensity(observed_pred, predictions, true_labels, title, max_yaxis): 
    """Plot KDE density plot for each classification on CM: TP, FP, TN, FP
    observed_pred: likelihood, comes from likelihood(model(input))
    predictions: class 0 or 1 predicted label, comes from model(input).loc.max(0)[1]
    true_labels: 0 or 1 true labels 
    title (str): plot title
    max_yaxis: max density (so all subplots on same y axis)
    """

    true_labels = true_labels.numpy()
    
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()
    
    # max_var = max(var_tp.max(), var_tn.max(), var_fp.max(), var_fn.max())
    # min_var = min(var_tp.min(), var_tn.min(), var_fp.min(), var_fn.min())
    max_y_lim = max_yaxis
    plt.figure(figsize=(10, 10))
    # to add same scale
    # bins = np.linspace(0, max(max(var_tp), max(var_tn), max(var_fp), max(var_fn)), 50)
    # bins = np.linspace(min_var, max_var, 50)
    plt.subplot(2, 2, 4)
    sns.histplot(var_tp, kde=True,color='green', bins=10, stat='density')
    plt.title('True Positives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 1)
    sns.histplot(var_tn, kde=True,color='blue', bins=10, stat='density')
    plt.title('True Negatives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 2)
    sns.histplot(var_fp, kde=True,color='red', bins=10, stat='density')
    plt.title('False Positives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 3)
    sns.histplot(var_fn, kde=True, color='orange', bins=10, stat='density')
    plt.title('False Negative', fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)
    
    plt.tight_layout()
    plt.suptitle(f'{title}', fontsize=16, y=1.05)
    plt.show();

def plot_prob_hist(probabilities, y_labels, title, bind_inhib): 
    """Histogram of prediction probabilities
    probabilities (tensor): sample from output distribution, and transform to probabilities
    y_labels: true labels 
    title: plot title
    bind_inhib (str): binding or inhibition for x axis label"""
    fig_width = 10
    fig_height = 8
    
    idx_1 = np.where(y_labels == 1)[0]
    idx_0 = np.where(y_labels == 0)[0]
    # Histogram predictions without error bars:
    fig, ax = plt.subplots(1,figsize=(fig_width, fig_height))
    ax.hist(probabilities.numpy()[1,][idx_1], histtype='step', linewidth=3, label='Binding')
    ax.hist(probabilities.numpy()[1,][idx_0], histtype='step', linewidth=3, label='No binding')
    ax.set_xlabel(f'Prediction ({bind_inhib} probability)')
    ax.set_ylabel('Number of compounds (in log scale)')
    plt.title(title, fontsize=24)
    plt.legend(fontsize=18)
    plt.yscale('log')
    plt.grid(True)
    plt.show(); 

def plot_swarmplot(predictions, true_labels, observed_pred, title):
    true_labels = true_labels.numpy()
    
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()

    data = {
        'Variance': np.concatenate([var_tp, var_tn, var_fp, var_fn]),
        'Category': ['TP'] * len(var_tp) + ['TN'] * len(var_tn) + ['FP'] * len(var_fp) + ['FN'] * len(var_fn)
    }

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Variance', data=df)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Variance')
    plt.show();


def probabilities_vs_var(true_labels, probabilities, observed_pred,title, bind_inhib):
    """Scatter plot of probabilities vs variance
    probabilities: extracted from samples
    """
    idx_1 = np.where(true_labels == 1)[0]
    idx_0 = np.where(true_labels == 0)[0]
    fig_width = 10
    fig_height = 8
    fig, ax = plt.subplots(1,figsize=(fig_width, fig_height))
    ax.scatter(probabilities.numpy()[1,][idx_1],
               observed_pred.variance.numpy()[1,][idx_1],
               label=bind_inhib, marker='^', s=80, alpha=0.75)

    ax.scatter(probabilities.numpy()[1,][idx_0],
               observed_pred.variance.numpy()[1,][idx_0],
               label=f'No {bind_inhib}', marker='o', s=80, alpha=0.75)
    
    ax.set_xlabel(f'Prediction ({bind_inhib} probability)')
    ax.set_ylabel(f'{bind_inhib} variance')
    plt.title(title, fontsize=24)
    plt.legend(fontsize=18)
    
    plt.show();


def swarm_prob(model, x_input, true_labels, title):
    """Swarm plot of probabilities (I used it for the rf models)
    model: rf model
    x_input: x labels 
    true_labels: matching y labels"""
    predictions = model.predict(x_input)
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    prob = model.predict_proba(x_input)
    a = prob[true_pos, 1]
    b = prob[true_neg, 0]
    c = prob[false_pos, 1]
    d = prob[false_neg, 0]
    data = {
        'Probability': np.concatenate([a,b,c,d]),
        'Category': ['TP'] * len(a) + ['TN'] * len(b) + ['FP'] * len(c) + ['FN'] * len(d)
    }

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Probability', data=df)
    plt.title(title)
    plt.xlabel('Classification Type')
    plt.ylabel('Probability')
    plt.show();
    
    

def plot_prec_recall(true_labels, probabilities_class1, title):
    precision, recall, thresholds = precision_recall_curve(true_labels, probabilities_class1)
    plt.figure(figsize=(8,6))
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.title(title)
    plt.show();



def swarm_by_var_and_prob(predictions, true_labels, observed_pred, probabilities, title):
    true_labels = true_labels.numpy()
   
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()
    prob_class0 = probabilities.numpy()[0,]
    prob_class1 = probabilities.numpy()[1,]
    prob_tp = probabilities.numpy()[1,][true_pos]
    prob_tn = probabilities.numpy()[0,][true_neg]
    prob_fp = probabilities.numpy()[1,][false_pos]
    prob_fn = probabilities.numpy()[0,][false_neg]
    
    

    data = {
        'Variance': np.concatenate([var_tp, var_tn, var_fp, var_fn]),
        'Probability Class 0 or Class 1': np.concatenate([prob_tp, prob_tn, prob_fp, prob_fn]),
        'Category': ['TP'] * len(var_tp) + ['TN'] * len(var_tn) + ['FP'] * len(var_fp) + ['FN'] * len(var_fn)
    }


    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Variance', data=df,hue='Probability Class 0 or Class 1')
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Variance')
    plt.show();
        
    
                