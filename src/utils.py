from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
