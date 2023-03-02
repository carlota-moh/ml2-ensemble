from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, mean_absolute_error, 
    silhouette_score, 
    classification_report, 
    RocCurveDisplay,
    f1_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    accuracy_score
)

def fit_pca(df, n_components=2, preprocessor=None):
    """
    Fits a PCA model to the dataframe

    Parameters
    ----------
    df : DataFrame
        Dataframe to fit the PCA
    n_components : int, optional
        Number of components to use, by default 2
    
    Returns
    -------
    pca : PCA
        PCA model
    df_pca : np.array
        Array with the PCA transformed data
    """
    if preprocessor is not None:
        df = preprocessor.fit_transform(df)
        n_components = df.shape[1]
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    return pca, df_pca

def get_explained_variance(pca, plot=True):
    """
    Returns the explained variance of the PCA model as a DataFrame and plots it

    Parameters
    ----------
    pca : PCA
        PCA model
    
    Returns
    -------
    exp_var : float
        Explained variance
    """
    # Get the explained variance of the PCA as a DataFrame
    exp_variance = pd.DataFrame(
        data=pca.explained_variance_ratio_, 
        index = ['PC' + str(n_pca + 1) for n_pca in range(pca.n_components)], 
        columns=['Exp_variance']
    )
    exp_variance['cum_Exp_variance'] = exp_variance['Exp_variance'].cumsum()

    # Plot the 10 first components of the PCA and the proportion of variance explained
    if plot:
        data = exp_variance[:10].copy() if exp_variance.shape[0] > 10 else exp_variance.copy()
        sns.barplot(data=data, x=data.index, y='Exp_variance', color='gray')
        sns.lineplot(data=data, x=data.index, y='cum_Exp_variance', 
                    color='blue', label='Cumulative Proportion')
        plt.gca().set_title('Explained Variance of the PCA')
        plt.gca().set_ylabel('Proportion of Variance Explained')
        plt.gca().set_xlabel('Principal Component')
        plt.legend()
        plt.show()

    return exp_variance

def get_n_components(exp_variance, threshold):
    """
    Get the number of components that explain a given threshold of the variance

    Parameters
    ----------
    exp_variance : DataFrame
        Explained variance of the PCA
    threshold : float
        Threshold of the variance explained
    
    Returns
    -------
    n_components : int
        Number of components that explain the threshold
    """
    n_components = exp_variance[exp_variance['cum_Exp_variance'] >= threshold].index[0]
    n_components = int(n_components.replace('PC', ''))
    return n_components

def get_loadings(pca, colNames):
    """
    Returns the loadings of the PCA model as a DataFrame

    Parameters
    ----------
    pca : PCA
        PCA model
    colNames : list
        List with the names of the columns

    Returns
    -------
    loadings : DataFrame
        Loadings of the PCA model
    """
    # Get the loadings of the PCA as a DataFrame
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns = ['PC' + str(n_pca + 1) for n_pca in range(pca.n_components)], 
        index = colNames)
    return loadings

def plot_loadings(loadings, components):
    """
    Plots the loadings of the PCA model

    Parameters
    ----------
    loadings : DataFrame
        Loadings of the PCA model
    components : list
        List with the components to plot
    """
    if len(components) == 1:
        sns.barplot(data=loadings[components], x=loadings.index, y=components[0])
        plt.gca().set_ylabel('Loadings')
        plt.gca().set_xlabel('Features')
        plt.xticks(rotation=90)
        
    else:
        fig, axes = plt.subplots(len(components), 1, figsize=(16,9))
        for component, ax in zip(components, axes.ravel()):
            sns.barplot(data=loadings[[component]], x=loadings.index, y=component, color='gray', ax=ax)
            plt.xticks(rotation=90)
    plt.show()
    return

def get_classification_metrics(models: dict, X_test, y_test, metrics, X_train=None, y_train=None, X_test_pca = None, X_train_pca = None):
    """
    Function that returns a metric table for a list of Classification models. Including:
    - Brier Loss
    - Log Loss
    - Precision
    - Recall
    - F1-Score
    - Accuracy

    Parameters
    ----------
    models : dict
        Dictionary with the model name and the model
    X_test : array-like
        Test data
    y_test : array-like, 1d
        Test labels
    metrics: list of str
        List of metrics to calculate. Possible values are 'brier_loss', 'log_loss', 'precision', 'recall', 'f1_score', 'accuracy'
    X_train : array-like, optional
        Train data
    y_train : array-like, 1d, optional
        Train labels
    
    Returns
    -------
    metric_table : pandas DataFrame
        DataFrame with the metrics for each model
    """

    # Create a dictionary with the metrics
    scores = defaultdict(list)
    # Initialize the flags as false:
    train_flag = False
    pca_flag = False
    # Iterate over the models
    for name, model in models.items():
        if 'pca' in name.lower():
            pca_flag = True
            y_pred_pca = model.predict(X_test_pca)
            # Calculate the metrics for the train data
            if X_train is not None and y_train is not None:
                train_flag = True
                y_pred_train_pca = model.predict(X_train_pca)
        else:
            pca_flag = False
            y_pred = model.predict(X_test)
            # Calculate the metrics for the train data
            if X_train is not None and y_train is not None:
                train_flag = True
                y_pred_train = model.predict(X_train)
        scores["Classifier"].append(name)
        # Iterate over the metrics
        for metric_name in metrics:
            if pca_flag:
                # Calculate the metrics for the test data
                score = get_score(metric_name, y_test, y_pred_pca)
                # Append the score to the scores dictionary
                scores[f"{metric_name.capitalize().replace('_', ' ')}"].append(score)
                if train_flag:
                    # Calculate the metrics for the train data
                    score_train = get_score(metric_name, y_train, y_pred_train_pca)
                    # Append the score to the scores dictionary
                    scores[f"{metric_name.capitalize().replace('_', ' ')} (Train)"].append(score_train)
            else:    
                # Calculate the metrics for the test data
                score = get_score(metric_name, y_test, y_pred)
                # Append the score to the scores dictionary
                scores[f"{metric_name.capitalize().replace('_', ' ')}"].append(score)
                if train_flag:
                    # Calculate the metrics for the train data
                    score_train = get_score(metric_name, y_train, y_pred_train)
                    # Append the score to the scores dictionary
                    scores[f"{metric_name.capitalize().replace('_', ' ')} (Train)"].append(score_train)

    # Create a DataFrame with the metrics
    metric_table = pd.DataFrame(scores).set_index("Classifier")
    # Round the values of the DataFrame
    metric_table.round(decimals=3)

    return metric_table

def get_score(metric_name, y_true, y_pred):
    """
    A nested function to calculate different scores based on the metric name

    Parameters
    ----------
    metric_name : str
        The name of the metric to calculate
    y_true : array-like, 1d
        The true labels
    y_pred : array-like, 1d
        The predicted labels

    Returns
    -------
    score : float
        The score for the given metric
    """
    metric_funcs = {
        'brier_loss':brier_score_loss, 
        'precision':precision_score, 
        'recall':recall_score, 
        'f1_score':f1_score, 
        'accuracy':accuracy_score
    }
    metric_func = metric_funcs.get(metric_name, precision_score)
    score = metric_func(y_true, y_pred)
    return score