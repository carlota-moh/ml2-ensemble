from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    brier_score_loss,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.calibration import (
    calibration_curve,
    CalibratedClassifierCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Constants
# Metrics for the summary function
METRIC_FUNCS_SUMMARY = {
    'Confusion matrix': confusion_matrix,
    'Confusion matrix normalized': confusion_matrix,
    'Accuracy': accuracy_score,
    'Balanced accuracy': balanced_accuracy_score, 
    'Precision': precision_score,
    'Recall': recall_score,
    'F1-Score': f1_score,
    'ROC-AUC': roc_auc_score
}

# Metrics for the model comparison function
METRIC_FUNCS_COMPARISON = {
    'Accuracy':accuracy_score,
    'Balanced accuracy': balanced_accuracy_score, 
    'Brier Loss':brier_score_loss, 
    'Precision':precision_score, 
    'Recall':recall_score, 
    'F1-Score': f1_score, 
    'ROC-AUC': roc_auc_score
}


def preprocess_data(df, target):
    """
    Preprocess the data by applying the following steps:
    1. Drop the target column
    2. Split the data into numeric and categorical columns
    3. Define the preprocessing pipelines for each column type
    4. Apply the pipelines to the data
    5. Combine the preprocessed numeric and categorical data
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess
    target : str
        The name of the target column
    
    Returns
    -------
    preprocessed_df : pd.DataFrame
        The preprocessed data
    y : pd.Series
        The target column
    """
    # Divide the data into X and y
    X, y = df.drop(target, axis=1), df[target]
    # Obtain the numeric and categorical columns in the data
    numeric_columns = X.select_dtypes(include=np.number).columns.to_list()
    categorical_columns = X.select_dtypes(include='category').columns.to_list()
    
    # Define the individual Pipelines
    std_scaler = StandardScaler()
    one_hot = OneHotEncoder(handle_unknown='ignore', drop='first')
    # Define the numeric and categorical preprocessing pipeline
    numeric_pipeline = Pipeline([('std_scaler', std_scaler)])
    categorical_pipeline = Pipeline([('one_hot', one_hot)])
    # Define the full preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_columns),
        ('categorical', categorical_pipeline, categorical_columns)
    ])

    # Apply the Pipeline to the data
    preprocessed_data = preprocessor.fit_transform(X)
    # Get the feature names after one-hot encoding
    feature_names = preprocessor.named_transformers_['categorical'].\
        named_steps['one_hot'].get_feature_names(categorical_columns).tolist()
    feature_names += numeric_columns

    # Create a DataFrame with the preprocessed data
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
    
    # Return the preprocessed data and the target
    return preprocessed_df, y


def get_metrics_summary_model(model, model_name, data_dict, threshold=0.5):
    """
    Prints a summary of the metrics of a model for a given threshold
    and returns them in a DataFrame

    Parameters
    ------------
    model : sklearn model
        The model to evaluate
    data_dict : dict of pd.DataFrame
        A dictionary containing the dataframes for train, test and validation sets (if any)
    threshold : float, optional
        The threshold to use to convert the probabilities to binary values, by default 0.5
    labels_sets : list, optional
        The labels to use for the train and test sets, by default ['TRAIN', 'TEST']

    Returns
    -----------
    pd.DataFrame
        A DataFrame with the metrics (accuracy, balanced accuracy, precision, recall, f1-score, roc-auc)

    Limitations
    -----------
    The data_dict dictionary should have the following structure, with the keys being the set names
    and the values being dictionaries with the keys 'data' and 'target' containing the data and target.

    data_dict = {
        'train': {
            'data': X_train,
            'target': y_train
        },
        'test': {
            'data': X_test,
            'target': y_test
        },
        'validation': {
            'data': X_val,
            'target': y_val
        }
    }

    """

    # Get the predicted probabilities and predictions
    predictions_dict = {}
    for dataset_type in data_dict.keys():
        # Get the predicted probabilities
        predictions_proba = model.predict_proba(data_dict[dataset_type]['data'])[:, 1]
        # Get the predictions and store them in a dictionary
        predictions = np.array([1 if i > threshold else 0 for i in predictions_proba])
        predictions_dict[dataset_type] = predictions

    # Calculate and print the metrics for each set
    metrics_dict = {
        'Model': model_name,
        'Threshold': threshold
    }
    # Iterate over the metrics
    for metric in METRIC_FUNCS_SUMMARY.keys():
        # Print a separator
        print('='*10)
        # Iterate over the sets
        for dataset_type, predictions in predictions_dict.items():
            # Calculate the metric for the set
            if metric == 'Confusion matrix':
                # Calculate the confusion matrix 
                metric_score = confusion_matrix(
                    data_dict[dataset_type]['target'], predictions
                )
                print(f'{metric} {dataset_type}: \n{metric_score}')
            elif metric == 'Confusion matrix normalized':
                # Calculate the normalized confusion matrix and round the values
                metric_score = confusion_matrix(
                    data_dict[dataset_type]['target'], predictions, normalize='true'
                )
                # Round it to 5 decimals
                metric_score = np.round(metric_score, 5)
                print(f'{metric} {dataset_type}: \n{metric_score}')
            else:
                # Calculate the other metrics
                metric_score = METRIC_FUNCS_SUMMARY[metric](
                    data_dict[dataset_type]['target'], predictions
                )
                print(f'{metric} {dataset_type}: {metric_score}')
                # Store the metric in the dictionary
                metrics_dict[f'{metric} {dataset_type}'] = metric_score

    # Return the metrics as a DataFrame
    return pd.DataFrame(metrics_dict, index=[0])


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
                score = get_score(metric_name, y_test, y_pred_pca, METRIC_FUNCS_COMPARISON)
                # Append the score to the scores dictionary
                scores[f"{metric_name.capitalize().replace('_', ' ')}"].append(score)
                if train_flag:
                    # Calculate the metrics for the train data
                    score_train = get_score(metric_name, y_train, y_pred_train_pca, METRIC_FUNCS_COMPARISON)
                    # Append the score to the scores dictionary
                    scores[f"{metric_name.capitalize().replace('_', ' ')} (Train)"].append(score_train)
            else:    
                # Calculate the metrics for the test data
                score = get_score(metric_name, y_test, y_pred, METRIC_FUNCS_COMPARISON)
                # Append the score to the scores dictionary
                scores[f"{metric_name.capitalize().replace('_', ' ')}"].append(score)
                if train_flag:
                    # Calculate the metrics for the train data
                    score_train = get_score(metric_name, y_train, y_pred_train, METRIC_FUNCS_COMPARISON)
                    # Append the score to the scores dictionary
                    scores[f"{metric_name.capitalize().replace('_', ' ')} (Train)"].append(score_train)

    # Create a DataFrame with the metrics
    metric_table = pd.DataFrame(scores).set_index("Classifier")
    # Round the values of the DataFrame
    metric_table.round(decimals=3)

    return metric_table


def get_score(metric_name, y_true, y_pred, metric_funcs):
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
    metric_func = metric_funcs.get(metric_name, precision_score)
    score = metric_func(y_true, y_pred)
    return score


def plot_calibration_curves_comparison(models: dict, X_test, y_test):
    """
    Plots the calibration curve and histograms for multiple models using Plotly.
    The different models are plotted in different colors and can be turned on and off by clicking on the legend.

    Parameters
    ----------
    models : dict
        A dictionary of the models to plot. Each key should be a string with the name of the model,
        and each value should be an estimator with a `predict_proba` method that takes `X_test` as input.
    X_test : array-like
        Test data
    y_test : array-like, 1d
        Test labels

    Returns
    -------
    None
    """
    # Define the colors for the plots
    colors = px.colors.qualitative.Dark2

    # Create a subplot for the calibration curve
    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title_text="Calibration Curves for the models")
    # Make sure the x and y axes are the same scale
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    # Plot the calibration curves for the models
    for i, (name, model) in enumerate(models.items()):
        # Get the predicted probabilities for the test data
        y_prob = model.predict_proba(X_test)[:, 1]
        # Get the true and predicted probabilities for the calibration curve
        true_prob, pred_prob = calibration_curve(y_test, y_prob, n_bins=10)
        # Plot the calibration curve
        trace = go.Scatter(x=pred_prob, y=true_prob, mode='lines+markers', name=name, line=dict(color=colors[i]))
        fig.add_trace(trace, row=1, col=1)

    # Add a diagonal line going through y=x
    diag_trace = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideal', line=dict(color='black', dash='dash'))
    fig.add_trace(diag_trace, row=1, col=1)

    # Add the x and y axis labels
    fig.update_xaxes(title_text='Predicted probability', row=1, col=1)
    fig.update_yaxes(title_text='True probability', row=1, col=1)
    # Add the legend and set the figure size
    fig.update_layout(showlegend=True, legend_title_text='Model', width=800, height=650)
    fig.show()
    return


def plot_predicted_probabilities_comparison(models, X_train, y_train, X_test, y_test):
    """
    Plots the predicted probabilities for each model in the dictionary. 
    Each model will have two plots, one for the training set and one for the test set.
    On each plot, the predicted probabilities are plotted for the positive and the negative class
    (positive in orange, negative in blue)

    Parameters
    ----------
    models : dict
        Dictionary of models to plot. The key is the name of the model and the value is the model object
    X_train : array-like
        Training data
    y_train : array-like, 1d
        Training labels
    X_test : array-like
        Test data
    y_test : array-like, 1d
        Test labels
    """
    # Create the figure and axes
    fig, axs = plt.subplots(len(models), 2, figsize=(10, len(models)*4.75))

    # Loop over each model and create the plots
    for i, (name, model) in enumerate(models.items()):
        # Obtain the predicted probabilities for the training set
        y_train_pred = model.predict_proba(X_train)[:, 1]
        neg_train_idx = np.where(y_train == 0)[0]
        pos_train_idx = np.where(y_train == 1)[0]
        # Convert the predicted probabilities into a histogram for the negative and positive class
        neg_train_counts, _ = np.histogram(y_train_pred[neg_train_idx], bins=np.linspace(0, 1, 11))
        pos_train_counts, _ = np.histogram(y_train_pred[pos_train_idx], bins=np.linspace(0, 1, 11))
        # Calculate the percentage of the positive and negative class in each bin
        neg_train_perc = neg_train_counts / len(neg_train_idx) * 100
        pos_train_perc = pos_train_counts / len(pos_train_idx) * 100

        # Plot the training set histogram for each class
        axs[i, 0].plot(np.linspace(0, 1, 10), neg_train_perc, '-o', label='Negative')
        axs[i, 0].plot(np.linspace(0, 1, 10), pos_train_perc, '-o', label='Positive')
        # Set the labels and title
        axs[i, 0].set_xlabel('Predicted probability')
        axs[i, 0].set_ylabel('Percentage')
        axs[i, 0].set_ylim([0, 100])
        axs[i, 0].set_title(f'{name} (train set)')
        # Add the legend
        axs[i, 0].legend()

        # Obtain the predicted probabilities for the test set
        y_test_pred = model.predict_proba(X_test)[:, 1]
        neg_test_idx = np.where(y_test == 0)[0]
        pos_test_idx = np.where(y_test == 1)[0]
        # Convert the predicted probabilities into a histogram for the negative and positive class
        neg_test_counts, _ = np.histogram(y_test_pred[neg_test_idx], bins=np.linspace(0, 1, 11))
        pos_test_counts, _ = np.histogram(y_test_pred[pos_test_idx], bins=np.linspace(0, 1, 11))
        # Calculate the percentage of the positive and negative class in each bin
        neg_test_perc = (neg_test_counts / len(neg_test_idx)) * 100
        pos_test_perc = (pos_test_counts / len(pos_test_idx)) * 100

        # Plot the training set histogram for each class
        axs[i, 1].plot(np.linspace(0, 1, 10), neg_test_perc, '-o', label='Negative')
        axs[i, 1].plot(np.linspace(0, 1, 10), pos_test_perc, '-o', label='Positive')
        # Set the labels and title
        axs[i, 1].set_xlabel('Predicted probability')
        axs[i, 1].set_ylabel('Percentage')
        axs[i, 1].set_ylim([0, 100])
        axs[i, 1].set_title(f'{name} (test set)')
        # Add the legend
        axs[i, 1].legend()

    # Set the plot title
    fig.suptitle('Predicted probabilities for each model and class for test and train set', fontsize=16)
    # Set the spacing between the plots
    plt.tight_layout()
    # Add some space between the title and the plots
    plt.subplots_adjust(top=0.96)
    # Show the plot
    plt.show()
    return