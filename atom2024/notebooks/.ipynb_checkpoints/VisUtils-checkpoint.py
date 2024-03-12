import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
import pandas as pd

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
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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
    plt.title(f'Inhibition KDE Variances for Each Class - {title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'nek2_inhib_kde_plot_{title}.png')
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