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

from imblearn.over_sampling import (
    SMOTE, 
    SMOTENC
)

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


def column_encoder(X_change, X_ref, custom_encoding=None):
    """
    Function used for applying specific column 
    encoding following the guidelines of another
    DataFrame.

    Parameters
    ----------
    X_change: pd.DataFrame
        DataFrame to encode
    X_ref: pd.DataFrame
        DataFrame taken as reference
    custom_encoding: dict, optional
        A dictionary containing custom encoding rules for specific columns. 
        Keys should be column names and values should be the data type to encode to.

    Returns
    -------
    encoded_df: pd.DataFrame
        A new DataFrame with the same columns as X_change but with the data types matched 
        to the encoding rules in X_ref or custom_encoding.
    """

    # Validate inputs
    if not isinstance(X_change, pd.DataFrame):
        raise TypeError("X_change must be a Pandas DataFrame.")
    if not isinstance(X_ref, pd.DataFrame):
        raise TypeError("X_ref must be a Pandas DataFrame.")
    if custom_encoding is not None and not isinstance(custom_encoding, dict):
        raise TypeError("custom_encoding must be a dictionary.")
    
    # Check for missing columns
    missing_cols = set(X_change.columns) - set(X_ref.columns)
    if missing_cols:
        raise ValueError(f"The following columns are missing from X_ref: {missing_cols}")
    
    # Apply custom encoding rules
    encoding = {}
    if custom_encoding is not None:
        encoding.update(custom_encoding)
    
    # Apply reference encoding rules
    for col in X_ref.columns:
        dtype = X_ref[col].dtype
        encoding[col] = dtype
    
    # Encode columns in X_change
    encoded_df = X_change.astype(encoding)
    
    return encoded_df


def fix_class_imbalance(X_train, y_train):
    """
    Fix imbalanced classes prior to training. Applies SMOTE or SMOTENC depending on the
    presence of categorical variables in the data.

    Parameters
    ----------
    X_train: pd.DataFrame
        The training data
    y_train: pd.Series
        The target column
    
    Returns
    -------
    X_balanced: pd.DataFrame
        The training data with balanced classes
    y_balanced: pd.Series
        The target column with balanced classes
    """

    # Fix class imbalance using SMOTE or SMOTENC
    cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
    cat_ids = [X_train.columns.get_loc(col) for col in cat_cols]
    # Use SMOTENC if categorical columns are present
    if len(cat_cols) >= 1:
        imbalance_fixer = SMOTENC(random_state=2022, categorical_features=cat_ids)
        X_balanced, y_balanced = imbalance_fixer.fit_resample(X_train.values, y_train.values)

    # if no cat variables are present, use regular SMOTE
    else:
        imbalance_fixer = SMOTE(random_state=2022)
        X_balanced, y_balanced = imbalance_fixer.fit_resample(X_train.values, y_train.values)

    # Fix format
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    column_encoder(X_balanced, X_train)
    # Convert y_balanced to a Series
    y_balanced = pd.Series(y_balanced, name=y_train.name)
    return X_balanced, y_balanced

def preprocess_data(X, preprocessor=None):
    """
    Preprocess the data by applying the following steps:
    1. If there is no preprocessor, fit one
    2. Apply the preprocessor to the data
    3. If the data is a numpy array, convert it to a DataFrame
    4. Return the preprocessed data and the preprocessor
    
    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to preprocess
    
    Returns
    -------
    X_prep : pd.DataFrame
        The preprocessed data
    """
    # If there is no preprocessor, fit one
    if preprocessor is None:
        preprocessor = fit_encoder(X)

    # Apply the preprocessor to the data
    X_prep = preprocessor.transform(X)

    # If the data is a numpy array, convert it to a DataFrame
    if isinstance(X_prep, np.ndarray):
        # Get the numeric and categorical columns
        feature_names = X.select_dtypes(include=np.number).columns.to_list()
        categorical_columns = X.select_dtypes(include='category').columns.to_list()
        # Then get the categorical columns
        try:
            feature_names += preprocessor.named_transformers_['categorical'].\
                named_steps['one_hot'].get_feature_names(categorical_columns).tolist()
        except AttributeError as e:
            feature_names += preprocessor.named_transformers_['categorical'].\
                named_steps['one_hot'].get_feature_names_out(categorical_columns).tolist()
        
        # Create a DataFrame with the preprocessed data
        X_prep = pd.DataFrame(X_prep, columns=feature_names)
        
    return X_prep, preprocessor


def fit_encoder(X):
    """
    Fit encoder used for encoding categorical and scaling numerical variables

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to preprocess
    
    Returns
    -------
    preprocessor : sklearn.preprocessing.ColumnTransformer
        The fitted encoder
    """
    # Obtain the numeric and categorical columns in the data
    numeric_columns = X.select_dtypes(include=np.number).columns.to_list()
    categorical_columns = X.select_dtypes(include='category').columns.to_list()
    
    # Define the individual Pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop='if_binary',
        handle_unknown='ignore'
    )
    numeric_pipeline = Pipeline([('std_scaler', numeric_transformer)])
    categorical_pipeline = Pipeline([('one_hot', categorical_transformer)])

    # Define the full preprocessing pipeline    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_columns),
            ('categorical', categorical_pipeline, categorical_columns)
        ],
    )
    
    # Fit the preprocessor and return it
    return preprocessor.fit(X)


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


def get_misclassified_days(model, X, y):
    """
    Returns the misclassified days by the given model.

    Parameters
    ----------
    model : sklearn model
        Model to use to predict the data.
    X : array-like
        Data to predict.
    y : array-like
        Target to compare the predictions.
    
    Returns
    -------
    misclassified_days: array-like
        Array with the misclassified days.
    """
    # Reset X and y indexes
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    # Get the predictions of the model
    y_pred = model.predict(X)
    # Get the misclassified days using a mask
    misclassified_mask = y_pred != y
    # Get the misclassified days data
    misclassified_days_data = X.loc[y[misclassified_mask].index]
    # Return the misclassified days data
    return misclassified_days_data


def get_misclassified_days_dict(model, data_dict):
    """
    Returns the misclassified days by the given model for each dataset in the data_dict.

    Parameters
    ----------
    model : sklearn model
        Model to use to predict the data.
    data_dict : dict
        Dictionary with the data to predict and the target to compare the predictions.
    
    Returns
    -------
    misclassified_days_dict: dict
        Dictionary with the misclassified days for each dataset in the data_dict.
    """
    misclassified_days_dict = {}
    for dataset_name, dataset in data_dict.items():
        misclassified_days_dict[dataset_name] = get_misclassified_days(model, dataset['data'], dataset['target'])
    return misclassified_days_dict


def plot_column_errors(model_errors, model_name, col_name, axis_title, title):
    """
    Plots barplots for each dataset in the model_errors dictionary showing the occurrences of each day of the week.

    Parameters
    ----------
    model_errors : dict
        Dictionary containing dataframes with the model errors for each dataset.
    model_name : str
        Name of the model to plot.
    col_name : str
        Name of the column to study.
    axis_title : str
        Title of the x-axis.
    title : str
        Title of the plot.
    """
    # Get the number of subplots needed based on the number of datasets in the input dictionary
    n_subplots = len(model_errors[model_name])

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(5*n_subplots, 5))

    # Loop over each dataset in the input dictionary
    for i, (dataset_name, dataset_errors) in enumerate(model_errors[model_name].items()):
        # Get the columns for the category of interest
        categoy_cols = [col for col in dataset_errors.columns if col.startswith(col_name)]
        # Sum the occurrences of each weekday across all rows
        category_counts = dataset_errors[categoy_cols].sum()
        # Plot the barplot for the current dataset in the corresponding subplot
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[i])
        ax.set_title(f"{dataset_name} ({len(dataset_errors)} errors)")
        ax.set_xlabel(axis_title)
        ax.set_ylabel('Occurrences')
        # Rotate the xticks to make them more readable
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # Despine the plot
        sns.despine()
        
    # Set the title of the figure
    fig.suptitle(f"{title} '{model_name}'")

    # Show the figure
    plt.show()
    return

def get_top_n_days(errors_dict, model, col_name, n=3, set = 'Validation'):
    """
    Get the top n days with the most errors for the specified model and dataset.

    Parameters
    ----------
    errors_dict : dict
        Dictionary containing dataframes with the model errors for each dataset.
    model : str
        Name of the model to plot.
    col_name : str
        Name of the column to study.
    n : int, optional
        Number of days to return, by default 3
    set : str, optional
        Dataset to use, by default 'Validation'
    """
    dataset_errors = errors_dict[model][set]
    # Get the columns for the category of interest
    categoy_cols = [col for col in dataset_errors.columns if col.startswith(col_name)]
    # Sum the occurrences of each weekday across all rows
    category_counts = dataset_errors[categoy_cols].sum()
    # Convert the series to a dataframe
    category_counts = pd.DataFrame(category_counts, columns=['Error_count'])
    # Format category counts
    category_counts = (
        category_counts
        # Reset the index to make the category a column
        .reset_index()
        # Rename the index to Category
        .rename(columns={'index': 'Category'})
        # Sort the dataframe by the error count and return the top n
        .sort_values(by='Error_count', ascending=False)
        .head(n)
    )
    # Sort the dataframe by the error count and return the top n
    return category_counts

# Define a function that obtains the misclassified days of the top 3 days with the most errors
def get_misclassified_days_top_n(errors_dict, model, col_name, n=3, set = 'Validation'):
    """
    Get the misclassified days of the top n days with the most errors for the specified model and dataset.

    Parameters
    ----------
    errors_dict : dict
        Dictionary containing dataframes with the model errors for each dataset.
    model : str
        Name of the model to plot.
    col_name : str
        Name of the column to study.
    n : int, optional
        Number of days to return, by default 3
    set : str, optional
        Dataset to use, by default 'Validation'
    """
    # Get the top n days with the most errors
    top_n_days = get_top_n_days(errors_dict, model, col_name, n, set)
    # Get the misclassified days for the model
    misclassified_days = errors_dict[model][set]
    # Get the misclassified days for the top n days whose error count is greater than 0
    days = top_n_days['Category'][top_n_days['Error_count'] > 0].tolist()
    misclassified_days_top_n = misclassified_days[misclassified_days[days].sum(axis=1) > 0]
    # Return the misclassified days for the top n days
    return days, misclassified_days_top_n

def compare_missclasified_days_top_n(X_train, errors_dict, model, col_name, n=3, set = 'Validation'):
    """
    Compare the misclassified days of the top n days with the average hours for the specified model and dataset.
    """
    # CR00 to CR23 are values for the 24 hours of the day
    hours = [f'CR{i:02d}' for i in range(24)]
    # Get the misclassified days for the top n days
    days, misclassified_days_top_n = get_misclassified_days_top_n(errors_dict, model, col_name, n, set)
    
    # Create a figure with as many vertical plots as there are days
    fig, axes = plt.subplots(nrows=len(days), ncols=1, figsize=(12, 6 * len(days)))
    for day, ax in zip(days, axes):
        # Get the mean value for each hour for the day
        hours_avg_day = X_train.loc[X_train[day] == 1, hours].mean()
        # Get the mean value for each hour for the misclassified days
        hours_error_day = misclassified_days_top_n.loc[misclassified_days_top_n[day] == 1, hours].mean()
        # Plot the mean values for each hour as a time series
        hours_avg_day.plot(ax = ax, label=f'Average for {day}', color='blue')
        hours_error_day.plot(ax = ax, label='Misclassified', color='red')
        # Put a horizontal line at 0 to make it easier to see the values
        ax.axhline(0, color='black', linestyle='--')
        # Make the y-axis go from the biggest out of the minimum or maximum value in absolute value
        max_y = max(hours_avg_day.abs().max(), hours_avg_day.abs().min(), hours_error_day.abs().max(), hours_error_day.abs().min())
        plt.ylim(-max_y, max_y)
        # Despine the plot
        sns.despine()
        # Change the x-axis ticks to be the hour of the day
        plt.xticks(range(24), range(24))
        # Add a legend outside the plot for this axis
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # Add a label to the y-axis
        ax.set_ylabel('Average value')
        # Add a label to the x-axis
        ax.set_xlabel('Hour of the day')

    # Add a title to the figure
    fig.suptitle(f'Average values for the hours of the day for {model} for the top {n} days with the most errors')
    plt.show()
    return